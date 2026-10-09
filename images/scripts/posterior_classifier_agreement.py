"""
Track the online class posterior p(c | x_t) while generating with the unconditional model, and
measure how well it predicts the class of the final image, taken as the prediction of a
pretrained ViT classifier.

sampler_with_posterior scores every sampled transition x_t -> x_next under each of the K
class-conditional models and under the unconditional model (sampling.uncond_run_name, if set),
and accumulates the per-step log-ratios from a uniform prior:

    log p(c | x_t) = log(1/K) + temp * sum_s [log q(x_next | x_s, c) - log q(x_next | x_s, uncond)].

Neither temp nor an offset changes the argmax of the posterior, since both act on all classes
alike: agreement with the ViT is temp-invariant and temp only affects calibration. The raw
log-ratios are therefore saved, and any temp is applied afterwards (temp sweep of the NLL below).

Reads:  logs/<run>/checkpoints/checkpoint_<k>/ema_model.safetensors (and <uncond_run> if set)
Writes: logs/Results/<run>/  (suffix: _ckpt<k>_guid<g>_steps<n>_<scheduler>_N<N>_<kernel>[_normbymask])
          classifier_accuracy_*.csv            per sample: target class, posterior and ViT predictions
          agreement_vs_time_*.csv              argmax-posterior / ViT agreement at t = 0, 0.1, ..., 1
          auc_vs_time_*.csv,                   macro and per-class one-vs-rest AUC over time, scored
          auc_softmax_tau<tau>_vs_time_*.csv     with the log-ratios or the softmax posterior at tau
          cum_logratios_t_*.pt                 [T, N, K] cumulative log-ratios, ViT predictions, t_next
          temp_sweep_*.csv                     NLL and agreement of the final posterior per temp
          macro_auc_*.csv                      one-vs-rest AUC of the final posterior
          roc_<score>_class<c>_*.csv           ROC curve for sampling.not_class
        logs/Figures/<run>/classifier_accuracy/guid_<g>/
          plots of the above, confusion matrix, class distribution, agreement vs. posterior margin,
          posterior trajectories and image trajectories of the first samples
Number of samples: sampling.num_samples.
"""
import os
import math
import hydra
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from datasets import load_from_disk
from omegaconf import DictConfig, OmegaConf
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.metrics import roc_curve, auc, roc_auc_score

from vision_model.dataset import DatacollatorImages
from vision_model.model.dit_model import DiTConfig
from discrete_flow_matching.sampling import sampler_with_posterior
from discrete_flow_matching.noise_scheduler import noising_images
from discrete_flow_matching.utils import set_all_seeds

from scripts.image_utils import (
    DATASET_CLASS_NAMES,
    PAPER_RC_PARAMS,
    tokens_to_rgb,
    tokens_to_display,
    get_classifier,
    load_dit_checkpoint,
    get_class_spectrum_colors,
    plot_and_print_metrics_samples_images,
    plot_agreement_vs_time,
    plot_auc_vs_time,
    plot_temp_sweep,
    plot_roc_curves,
    plot_confusion_matrix,
    plot_class_distribution,
    plot_posterior_trajectory,
    plot_agreement_vs_margin,
)

plt.rcParams.update(PAPER_RC_PARAMS)

# Analysis settings. Neither OFFSET nor temp changes the argmax (both act uniformly on all classes);
# temp only affects calibration, hence the NLL sweep (see module docstring).
DEFAULT_TEMP      = 0.5                           # temp of the sampler's posterior (post_probs)
OFFSET            = 0.0
TEMP_SWEEP        = torch.linspace(0.1, 1.0, 10)
TEMP_FOR_VIS      = 0.2                           # temp of the posterior trajectory figures
N_TRAJ            = 4                             # number of trajectory figures
ROC_SOFTMAX_TEMPS = [0.1, 0.5, 1.0]
AUC_SOFTMAX_TAUS  = [0.01, 0.05, 0.1, 0.5]


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run_classifier_accuracy(cfg: DictConfig):
    set_all_seeds(1)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name         = cfg_dict["sampling"]["run_name"]
    uncond_run_name  = cfg_dict["sampling"].get("uncond_run_name", None)
    checkpoints      = cfg_dict["sampling"]["to_evaluate_checkpoints"]
    diffusion_type   = cfg_dict["noise"]["type"]
    scheduler_type   = cfg_dict["noise"]["scheduler_type"]
    polynomial_power = cfg_dict["noise"]["polynomial_power"]
    t_max_train      = cfg_dict["trainer"]["t_max_train"]
    t_min_train      = cfg_dict["trainer"]["t_min_train"]
    num_steps        = cfg_dict["sampling"]["num_steps"]
    guid_scale       = cfg_dict["sampling"]["guid_scale"]
    num_samples      = cfg_dict["sampling"]["num_samples"]
    cache_dir        = cfg_dict["dataset"]["path"]
    dataset_name     = cfg_dict["dataset"]["name"]
    class_names      = DATASET_CLASS_NAMES.get(dataset_name, None)
    batch_size       = cfg_dict["sampling"]["batch_size"]
    use_ancestral_transition_probs = cfg_dict["sampling"]["use_ancestral_sampling_transition_likelihoods"]
    normalize_by_unmask_count      = cfg_dict["sampling"]["normalize_by_unmask_count"]

    kernel_suffix = '_anc_trans_probs' if use_ancestral_transition_probs else '_ode_trans_probs'
    norm_suffix   = '_normbymask' if normalize_by_unmask_count else ''
    file_suffix   = kernel_suffix + norm_suffix

    if diffusion_type == 'masking':
        cfg_dict["model"]["vocab_size"] += 1
    vocab_size = cfg_dict["model"]["vocab_size"]

    print(f'diffusion_type={diffusion_type}, scheduler={scheduler_type}, num_steps={num_steps}, guid_scale={guid_scale}')
    print(f'Generating {num_samples} samples in batches of {batch_size} ...')
    if uncond_run_name is not None:
        print(f"  Using separately-trained unconditional model '{uncond_run_name}' for every "
              f"uncond-token forward pass (CFG uncond branch + posterior uncond baseline), "
              f"regardless of guid_scale={guid_scale}.")

    model_config = DiTConfig(
        hidden_size=cfg_dict["model"]["hidden_size"],
        depth=cfg_dict["model"]["depth"],
        num_heads=cfg_dict["model"]["num_heads"],
        frequency_embedding_size=cfg_dict["model"]["frequency_embedding_size"],
        input_size=cfg_dict["model"]["input_size"],
        in_channels=cfg_dict["model"]["in_channels"],
        vocab_size=vocab_size,
    )

    output_fig_dir = f"logs/Figures/{run_name}/classifier_accuracy/guid_{guid_scale}"
    os.makedirs(output_fig_dir, exist_ok=True)
    results_dir = f"logs/Results/{run_name}"
    os.makedirs(results_dir, exist_ok=True)

    print(f'Loading data from: {cache_dir}')
    ds = load_from_disk(cache_dir)

    collator = DatacollatorImages(
        return_labels=True,
        diffusion_type=diffusion_type,
        scheduler_type=scheduler_type,
        polynomial_power=polynomial_power,
        t_max_train=t_max_train,
        t_min_train=t_min_train,
        vocab_size=vocab_size,
    )

    dataloader = DataLoader(
        ds["test"],
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=False,
    )

    classifier = get_classifier(dataset_name, device=device)
    num_classes = (len(class_names) - 1) if class_names is not None else 10
    if classifier.num_classes != num_classes:
        raise ValueError(
            f"Classifier for '{dataset_name}' has {classifier.num_classes} classes, "
            f"but this run expects {num_classes}. Check _CLASSIFIER_DEFAULTS "
            f"in image_utils.py / DATASET_CLASS_NAMES in this script."
        )

    # Only used for its shape: at t=0 the image is fully noised anyway.
    c_batch, x_t_batch, t_batch, y_batch = next(iter(dataloader))
    y_full = torch.where(y_batch == -100, x_t_batch, y_batch).to(device)
    B_data, C, H, W = y_full.shape
    if B_data < batch_size:
        raise ValueError(f"sampling.batch_size={batch_size} exceeds the dataloader batch ({B_data}); lower it.")

    for checkpoint_number in checkpoints:
        print(f"\n{'='*40}\nCheckpoint {checkpoint_number}\n{'='*40}")

        cond_model = load_dit_checkpoint(run_name, checkpoint_number, model_config, device)
        uncond_model = cond_model
        if uncond_run_name is not None:
            uncond_model = load_dit_checkpoint(uncond_run_name, checkpoint_number, model_config, device)

        num_batches = math.ceil(num_samples / batch_size)
        print(f"Running sampler_with_posterior over {num_batches} batches of up to {batch_size} "
              f"({num_classes + 1} forward passes/step — this is slow) ...")
        if use_ancestral_transition_probs:
            print('  Using ancestral (MD4 CTMC) transitions for posterior scoring.')
        else:
            print('  Using DFM-ODE transitions for posterior scoring.')
        if normalize_by_unmask_count:
            print('  Normalizing each step\'s posterior update by its unmasking-event count.')

        # post_probs[i] is the posterior after the step ending at t_next = t_axis[i+1].
        T_TARGETS = [round(0.1 * i, 1) for i in range(11)]
        _t_next_values = torch.linspace(0, 1, num_steps)[1:]
        t_target_indices = [int(torch.argmin((_t_next_values - tt).abs())) for tt in T_TARGETS]
        t_target_actual  = [round(_t_next_values[idx].item(), 4) for idx in t_target_indices]
        agree_count_per_t = torch.zeros(len(T_TARGETS), dtype=torch.float64)

        records = []
        cum_logratios_list, vit_pred_list = [], []
        cum_logratios_t_list = []
        num_generated = 0
        for batch_idx in range(num_batches):
            this_batch_size = min(batch_size, num_samples - num_generated)

            gen_class = cfg_dict["sampling"].get("gen_class", None)
            if gen_class is None:
                c_start = torch.randint(0, num_classes, (this_batch_size,), dtype=torch.long, device=device)
            else:
                c_start = torch.full((this_batch_size,), gen_class, dtype=torch.long, device=device)
            c_uncond = torch.full((this_batch_size,), num_classes, dtype=torch.long, device=device)

            t_start = torch.zeros(this_batch_size, device=device)
            x_in, _ = noising_images(
                y_full[:this_batch_size], t_start, vocab_size,
                diffusion_type=diffusion_type,
                scheduler_type=scheduler_type,
                polynomial_power=polynomial_power,
            )

            final_xs, xs, post_probs, unnorm_post_probs, raw_logratios = sampler_with_posterior(
                cond_model, c_start, c_uncond, num_classes,
                guid_scale=guid_scale,
                x_in=x_in,
                uncond_model=uncond_model,
                diffusion_type=diffusion_type,
                scheduler_type=scheduler_type,
                polynomial_power=polynomial_power,
                num_steps=num_steps,
                vocab_size=vocab_size,
                batch_size=this_batch_size,
                C=C, H=H, W=W, temp=DEFAULT_TEMP, offset=OFFSET,
                use_ancestral_sampling_probs=use_ancestral_transition_probs,
                normalize_by_unmask_count=normalize_by_unmask_count,
            )

            dfm_probs = post_probs[-1]              # [B, K]
            dfm_pred  = dfm_probs.argmax(dim=-1)
            dfm_conf  = dfm_probs.max(dim=-1).values
            dfm_top2  = torch.topk(dfm_probs, k=2, dim=-1).values
            dfm_margin = dfm_top2[:, 0] - dfm_top2[:, 1]

            pil_imgs = [Image.fromarray(tokens_to_rgb(final_xs[i], vocab_size)) for i in range(this_batch_size)]
            vit_pred, vit_conf, _ = classifier.predict(pil_imgs)

            # The ViT prediction on the final image is the reference at every t.
            vit_pred_cpu = vit_pred.detach().cpu()
            for ti, idx in enumerate(t_target_indices):
                dfm_pred_t = post_probs[idx].argmax(dim=-1).detach().cpu()
                agree_count_per_t[ti] += (dfm_pred_t == vit_pred_cpu).double().sum().item()

            # raw_logratios [T, B, K] is temp/offset-independent, so any temp can be applied post-hoc.
            cum_logratios_list.append(raw_logratios.sum(dim=0).detach().cpu())  # [B, K]
            cum_logratios_t_list.append(torch.cumsum(raw_logratios, dim=0).detach().cpu())  # [T, B, K]
            vit_pred_list.append(vit_pred.detach().cpu())

            if batch_idx == 0:
                n_traj_actual        = min(N_TRAJ, this_batch_size)
                traj_raw_logratios   = raw_logratios[:, :n_traj_actual, :].detach().cpu()  # [T, n_traj, K]
                traj_final_xs        = final_xs[:n_traj_actual].detach().cpu()
                traj_xs              = xs[:, :n_traj_actual].detach().cpu()               # [T, n_traj, C, H, W]
                traj_vit_pred        = vit_pred[:n_traj_actual].detach().cpu()
                traj_vit_conf        = vit_conf[:n_traj_actual].detach().cpu()

            for i in range(this_batch_size):
                records.append({
                    "checkpoint": checkpoint_number,
                    "c_start": c_start[i].item(),
                    "dfm_pred": dfm_pred[i].item(),
                    "dfm_conf": dfm_conf[i].item(),
                    "dfm_margin": dfm_margin[i].item(),
                    "vit_pred": vit_pred[i].item(),
                    "vit_conf": vit_conf[i].item(),
                })

            num_generated += this_batch_size
            print(f"  [{num_generated}/{num_samples}] batch {batch_idx + 1}/{num_batches} done")

        df = pd.DataFrame(records)

        agree_rate    = (df["dfm_pred"] == df["vit_pred"]).mean()
        dfm_vs_target = (df["dfm_pred"] == df["c_start"]).mean()
        vit_vs_target = (df["vit_pred"] == df["c_start"]).mean()

        print(f"\n--- Checkpoint {checkpoint_number} summary (N={len(df)}) ---")
        print(f"  Online class posterior vs ViT classifier agreement: {agree_rate:.4f}")
        print(f"  Online class posterior vs nominal target class:     {dfm_vs_target:.4f}")
        print(f"  ViT classifier vs nominal target class:              {vit_vs_target:.4f}")

        csv_path = (f"{results_dir}/classifier_accuracy_ckpt{checkpoint_number}_guid{guid_scale}"
                    f"_steps{num_steps}_{scheduler_type}_N{len(df)}{file_suffix}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Saved per-sample results to {csv_path}")

        # --- Agreement vs. t ---
        agreement_vs_time = (agree_count_per_t / num_generated).tolist()
        print(f"\n  --- Agreement vs. diffusion time t (online class posterior vs ViT classifier"
              f"{', unmask-count normalised' if normalize_by_unmask_count else ''}) ---")
        for tt, ta, ag in zip(T_TARGETS, t_target_actual, agreement_vs_time):
            print(f"    t={tt:.1f} (actual t_next={ta:.3f}): agreement={ag:.4f}")

        time_df = pd.DataFrame({
            "t_target": T_TARGETS, "t_actual": t_target_actual,
            "agreement": agreement_vs_time, "n": num_generated,
        })
        time_csv_path = (f"{results_dir}/agreement_vs_time_ckpt{checkpoint_number}_guid{guid_scale}"
                          f"_steps{num_steps}_{scheduler_type}_N{num_generated}{file_suffix}.csv")
        time_df.to_csv(time_csv_path, index=False)
        print(f"  Saved agreement-vs-time results to {time_csv_path}")

        norm_note = ' (unmask-count normalised)' if normalize_by_unmask_count else ''
        time_fig_path = (f"{output_fig_dir}/ckpt_{checkpoint_number}_agreement_vs_time_steps{num_steps}"
                          f"_guid{guid_scale}_N{num_generated}{file_suffix}.png")
        plot_agreement_vs_time(T_TARGETS, agreement_vs_time, num_classes,
                               f"Agreement vs. t — ckpt {checkpoint_number}, N={num_generated}{norm_note}",
                               time_fig_path)
        print(f"  Saved agreement-vs-time plot to {time_fig_path}")

        # --- Macro one-vs-rest AUC vs. t ---
        # Label: ViT prediction on the final image. Scores: the unnormalised class-vs-uncond
        # log-odds ('unnorm') or the softmax posterior p(c|x_t) at several tau.
        # t=0 is prepended with AUC=0.5 (no evidence yet, all scores tied).
        cum_t_all   = torch.cat(cum_logratios_t_list, dim=1)  # [T, N, K]
        vit_pred_np = torch.cat(vit_pred_list, dim=0).numpy()  # [N]
        auc_t_axis  = np.concatenate([[0.0], _t_next_values.numpy()])  # [T+1]
        valid_classes = [c for c in range(num_classes)
                         if 0 < (vit_pred_np == c).sum() < len(vit_pred_np)]

        cum_t_path = (f"{results_dir}/cum_logratios_t_ckpt{checkpoint_number}_guid{guid_scale}"
                       f"_steps{num_steps}_{scheduler_type}_N{num_generated}{file_suffix}.pt")
        torch.save({"cum_logratios_t": cum_t_all, "vit_pred": torch.from_numpy(vit_pred_np),
                    "t_axis": torch.from_numpy(auc_t_axis[1:])}, cum_t_path)
        print(f"  Saved per-step cumulative log posterior [T={cum_t_all.shape[0]}, N, K] to {cum_t_path}")

        # tag -> (scores [T, N, K], filename stem, y-label, title tag)
        auc_time_variants = {
            "unnorm": (cum_t_all, "auc_vs_time", "unnormalised log posterior", "unnormalised"),
        }
        for tau in sorted({DEFAULT_TEMP, *AUC_SOFTMAX_TAUS}):
            auc_time_variants[f"softmax_tau{tau}"] = (
                torch.softmax(tau * cum_t_all, dim=-1), f"auc_softmax_tau{tau}_vs_time",
                rf"softmax posterior $p(c|x_t)$, $\tau$={tau}", f"softmax, tau={tau}")
        for tag, (score_tnk, stem, ylab, title_tag) in auc_time_variants.items():
            auc_class_arr = np.full((len(auc_t_axis), num_classes), np.nan)
            auc_class_arr[0, valid_classes] = 0.5
            for ti in range(score_tnk.shape[0]):
                scores_t = score_tnk[ti].numpy()  # [N, K]
                for c in valid_classes:
                    auc_class_arr[ti + 1, c] = roc_auc_score((vit_pred_np == c).astype(int), scores_t[:, c])
            auc_macro = np.nanmean(auc_class_arr, axis=1)

            auc_time_df = pd.DataFrame({
                "t": auc_t_axis,
                "auc": auc_macro, "n_classes_used": len(valid_classes), "n": num_generated,
            })
            for c in range(num_classes):
                auc_time_df[f"auc_class_{c}"] = auc_class_arr[:, c]
            auc_time_csv_path = (f"{results_dir}/{stem}_ckpt{checkpoint_number}_guid{guid_scale}"
                                  f"_steps{num_steps}_{scheduler_type}_N{num_generated}{file_suffix}.csv")
            auc_time_df.to_csv(auc_time_csv_path, index=False)
            print(f"  Macro one-vs-rest AUC vs t [{tag}]: t=0 -> {auc_macro[0]:.4f}, t=0.5 -> "
                  f"{auc_macro[int(np.argmin(np.abs(auc_t_axis - 0.5)))]:.4f}, t=1 -> {auc_macro[-1]:.4f} "
                  f"({len(valid_classes)}/{num_classes} classes)")
            print(f"  Saved AUC-vs-time results [{tag}] (macro + per-class) to {auc_time_csv_path}")

            auc_time_fig_path = (f"{output_fig_dir}/ckpt_{checkpoint_number}_{stem}_steps{num_steps}"
                                  f"_guid{guid_scale}_N{num_generated}{file_suffix}.png")
            plot_auc_vs_time(auc_t_axis, auc_class_arr, auc_macro, valid_classes, ylab,
                             f"AUC vs. t ({title_tag}) — ckpt {checkpoint_number}, N={num_generated}{norm_note}",
                             auc_time_fig_path)
            print(f"  Saved AUC-vs-time plot [{tag}] to {auc_time_fig_path}")

        # --- Temp sweep: NLL of the posterior at the ViT-predicted class (agreement is temp-invariant) ---
        cum_logratios_all = torch.cat(cum_logratios_list, dim=0)  # [N, K]
        vit_pred_all       = torch.cat(vit_pred_list, dim=0)       # [N]
        n_total            = cum_logratios_all.shape[0]

        sweep_records = []
        for temp_val in TEMP_SWEEP.tolist():
            probs = torch.softmax(temp_val * cum_logratios_all, dim=-1)
            pred  = probs.argmax(dim=-1)
            agreement_t = (pred == vit_pred_all).float().mean().item()
            nll_t = -torch.log(probs[torch.arange(n_total), vit_pred_all] + 1e-12).mean().item()
            sweep_records.append({"temp": temp_val, "agreement": agreement_t, "nll": nll_t})

        sweep_df = pd.DataFrame(sweep_records)
        best_temp = sweep_df.loc[sweep_df["nll"].idxmin(), "temp"]
        print(f"  Temp sweep: NLL ranges [{sweep_df['nll'].min():.4f}, {sweep_df['nll'].max():.4f}], "
              f"best temp={best_temp:.3f}; agreement is flat at {sweep_df['agreement'].mean():.4f} "
              f"(std={sweep_df['agreement'].std():.2e}, as expected -- see docstring).")

        sweep_csv_path = (f"{results_dir}/temp_sweep_ckpt{checkpoint_number}_guid{guid_scale}"
                           f"_steps{num_steps}_{scheduler_type}_N{n_total}{file_suffix}.csv")
        sweep_df.to_csv(sweep_csv_path, index=False)
        print(f"  Saved temp-sweep results to {sweep_csv_path}")

        sweep_fig_path = (f"{output_fig_dir}/ckpt_{checkpoint_number}_temp_sweep_steps{num_steps}"
                           f"_guid{guid_scale}_N{n_total}{file_suffix}.png")
        plot_temp_sweep(sweep_df["temp"], sweep_df["nll"], sweep_df["agreement"], best_temp,
                        f"Temp sweep — ckpt {checkpoint_number}, N={n_total}", sweep_fig_path)
        print(f"  Saved temp-sweep plot to {sweep_fig_path}")

        # --- Macro one-vs-rest AUC at t=1 (classes without both labels are skipped) ---
        macro_auc_records = []
        macro_auc_score_fns = {
            "unnorm": lambda c: cum_logratios_all[:, c].numpy(),
            f"softmax_tau{DEFAULT_TEMP}": lambda c: torch.softmax(DEFAULT_TEMP * cum_logratios_all, dim=-1)[:, c].numpy(),
        }
        for score_name, score_fn in macro_auc_score_fns.items():
            per_class_auc = {}
            for c in range(num_classes):
                label = (vit_pred_all == c).numpy().astype(int)
                if label.min() == label.max():
                    continue
                fpr, tpr, _ = roc_curve(label, score_fn(c))
                per_class_auc[c] = auc(fpr, tpr)
            macro_auc = float(np.mean(list(per_class_auc.values()))) if per_class_auc else float("nan")
            print(f"  Macro one-vs-rest AUC [{score_name}] over {len(per_class_auc)}/{num_classes} classes: {macro_auc:.4f}")
            for c, class_auc in per_class_auc.items():
                macro_auc_records.append({"score": score_name, "class": c, "auc": class_auc})
            macro_auc_records.append({"score": score_name, "class": "macro_avg", "auc": macro_auc})

        macro_auc_df = pd.DataFrame(macro_auc_records)
        macro_auc_csv_path = (f"{results_dir}/macro_auc_ckpt{checkpoint_number}_guid{guid_scale}"
                               f"_steps{num_steps}_{scheduler_type}_N{n_total}{file_suffix}.csv")
        macro_auc_df.to_csv(macro_auc_csv_path, index=False)
        print(f"  Saved macro one-vs-rest AUC results to {macro_auc_csv_path}")

        # --- ROC curves for sampling.not_class vs. ViT ground truth ---
        # 'unnorm' (class-vs-uncond log-odds, as used by the NOT gate) is tau-invariant;
        # the softmax posterior p(c|x_t) is not, since tau changes the ranking across samples.
        roc_class = int(cfg_dict["sampling"].get("not_class", 0))
        roc_label = (vit_pred_all == roc_class).numpy().astype(int)
        n_pos = int(roc_label.sum())

        softmax_cmap = plt.colormaps['Reds']
        roc_variants = {
            "unnorm": ("class-vs-uncond log-odds (unnormalised)",
                       cum_logratios_all[:, roc_class].numpy(), "tab:blue"),
        }
        for i, tau in enumerate(ROC_SOFTMAX_TEMPS):
            shade = softmax_cmap(0.4 + 0.5 * i / max(len(ROC_SOFTMAX_TEMPS) - 1, 1))
            roc_variants[f"softmax_tau{tau}"] = (
                f"class-vs-rest softmax posterior (tau={tau})",
                torch.softmax(tau * cum_logratios_all, dim=-1)[:, roc_class].numpy(),
                shade,
            )

        if roc_label.min() == roc_label.max():
            print(f"  Skipping ROC curves for class {roc_class}: only one label value present "
                  f"among {n_total} samples ({n_pos} positive) -- need both positive and negative examples.")
        else:
            roc_curves = []
            for variant, (variant_label, roc_score, color) in roc_variants.items():
                fpr, tpr, roc_thresholds = roc_curve(roc_label, roc_score)
                roc_auc = auc(fpr, tpr)
                print(f"  ROC [{variant}] ({variant_label}) for class {roc_class} vs ViT ground truth: "
                      f"AUC={roc_auc:.4f}  (n_pos={n_pos}/{n_total})")

                roc_df = pd.DataFrame({
                    "fpr":       np.concatenate([fpr, [np.nan]]),
                    "tpr":       np.concatenate([tpr, [np.nan]]),
                    "threshold": np.concatenate([roc_thresholds, [np.nan]]),
                })
                roc_csv_path = (f"{results_dir}/roc_{variant}_class{roc_class}_ckpt{checkpoint_number}_guid{guid_scale}"
                                  f"_steps{num_steps}_{scheduler_type}_N{n_total}_auc{roc_auc:.4f}{file_suffix}.csv")
                roc_df.to_csv(roc_csv_path, index=False)
                print(f"    Saved ROC curve data to {roc_csv_path}")

                legend_tag = "unnorm" if variant == "unnorm" else f"softmax τ={variant.split('tau')[1]}"
                roc_curves.append((fpr, tpr, f"{legend_tag} (AUC = {roc_auc:.3f})", color))

            class_label = class_names[roc_class] if class_names else str(roc_class)
            roc_fig_path = (f"{output_fig_dir}/ckpt_{checkpoint_number}_roc_class{roc_class}_steps{num_steps}"
                              f"_guid{guid_scale}_N{n_total}{file_suffix}.png")
            plot_roc_curves(roc_curves,
                            f"ROC — internal evidence for {class_label} vs ViT ground truth\n"
                            f"ckpt {checkpoint_number}, N={n_total} (n_pos={n_pos}), guid_scale={guid_scale}",
                            roc_fig_path)
            print(f"  Saved combined ROC curve plot to {roc_fig_path}")

        # --- Confusion matrix ---
        confusion = np.zeros((num_classes, num_classes), dtype=int)
        for _, row in df.iterrows():
            confusion[int(row["vit_pred"]), int(row["dfm_pred"])] += 1

        tick_labels = class_names[:num_classes] if class_names else [str(k) for k in range(num_classes)]

        fig_path = (f"{output_fig_dir}/ckpt_{checkpoint_number}_confusion_steps{num_steps}"
                    f"_guid{guid_scale}_N{len(df)}{file_suffix}.png")
        plot_confusion_matrix(confusion, tick_labels,
                              f"Confusion — ckpt {checkpoint_number}, agreement={agree_rate:.3f}, N={len(df)}",
                              fig_path)
        print(f"  Saved confusion matrix to {fig_path}")

        # --- Predicted class distribution ---
        dfm_counts = df["dfm_pred"].value_counts().reindex(range(num_classes), fill_value=0)
        vit_counts = df["vit_pred"].value_counts().reindex(range(num_classes), fill_value=0)
        dfm_pct = 100.0 * dfm_counts / len(df)
        vit_pct = 100.0 * vit_counts / len(df)

        dist_fig_path = (f"{output_fig_dir}/ckpt_{checkpoint_number}_class_distribution_steps{num_steps}"
                          f"_guid{guid_scale}_N{len(df)}{file_suffix}.png")
        plot_class_distribution(dfm_pct.values, vit_pct.values, tick_labels,
                                f"Predicted class distribution — ckpt {checkpoint_number}, N={len(df)}",
                                dist_fig_path)
        print(f"  Saved class-distribution plot to {dist_fig_path}")

        # --- Posterior trajectories of the first N_TRAJ samples, at TEMP_FOR_VIS ---
        traj_cumsum = torch.cumsum(traj_raw_logratios, dim=0)  # [T, n_traj, K]
        traj_post_log_likelihood = math.log(1.0 / num_classes) + TEMP_FOR_VIS * traj_cumsum
        traj_post_probs        = torch.softmax(traj_post_log_likelihood, dim=-1)  # [T, n_traj, K]
        traj_unnorm_post_probs = torch.exp(traj_post_log_likelihood)              # [T, n_traj, K]

        t_axis_traj  = _t_next_values  # traj_post_probs[i] is the posterior at t_next = t_axis[i+1]
        class_colors = get_class_spectrum_colors(num_classes)

        for traj_i in range(traj_post_probs.shape[1]):
            gen_class_idx  = traj_vit_pred[traj_i].item()
            gen_confidence = traj_vit_conf[traj_i].item()
            gen_name       = classifier.class_names[gen_class_idx]
            final_rgb      = tokens_to_display(traj_final_xs[traj_i], vocab_size)
            traj_tag       = f"traj {traj_i}  (classifier pred: {gen_name}, conf={gen_confidence:.2f})"
            traj_path      = (f'{output_fig_dir}/ckpt_{checkpoint_number}_posterior_traj{traj_i}_steps{num_steps}'
                              f'_guid{guid_scale}_temp{TEMP_FOR_VIS}')

            plot_posterior_trajectory(
                t_axis_traj, traj_post_probs[:, traj_i], final_rgb, tick_labels, class_colors,
                class_colors[gen_class_idx], f'Class posterior — {traj_tag}', r'$p(c|x_t)$ (with softmax)',
                f'{traj_path}{file_suffix}.png', legend_above=True,
            )
            plot_posterior_trajectory(
                t_axis_traj, traj_unnorm_post_probs[:, traj_i], final_rgb, tick_labels, class_colors,
                class_colors[gen_class_idx], f'Unnormalised class posterior — {traj_tag}', r'$p(c|x_t)$ (with exp)',
                f'{traj_path}_unnorm{file_suffix}.png',
            )

        print(f"  Saved {traj_post_probs.shape[1]} posterior trajectory figures "
              f"(TEMP_FOR_VIS={TEMP_FOR_VIS}) to {output_fig_dir}/")

        plot_and_print_metrics_samples_images(
            None, traj_xs, traj_final_xs, traj_vit_pred, vocab_size,
            output_fig_dir, checkpoint_number, 0,
            save_images=False, save_trajectories=True, save_num_switches_plot=False,
            add_fig_label=f'steps_{num_steps}_guid_{guid_scale}{file_suffix}',
            class_names=class_names,
        )

        # --- Agreement vs. confidence margin (top1 - top2) ---
        df["agree"] = df["dfm_pred"] == df["vit_pred"]
        bin_edges = np.linspace(0.0, 1.0, 11)
        df["margin_bin"] = pd.cut(df["dfm_margin"], bins=bin_edges, include_lowest=True)
        binned = df.groupby("margin_bin", observed=False)["agree"].agg(["mean", "count"])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        valid = binned["count"] > 0
        margin_fig_path = (f"{output_fig_dir}/ckpt_{checkpoint_number}_agreement_vs_margin_steps{num_steps}"
                            f"_guid{guid_scale}_N{len(df)}{file_suffix}.png")
        plot_agreement_vs_margin(bin_centers[valid], binned["mean"][valid], binned["count"][valid], num_classes,
                                 f"Agreement vs. DFM confidence — ckpt {checkpoint_number}, N={len(df)}",
                                 margin_fig_path)
        print(f"  Saved agreement-vs-margin plot to {margin_fig_path}")


if __name__ == "__main__":
    run_classifier_accuracy()
