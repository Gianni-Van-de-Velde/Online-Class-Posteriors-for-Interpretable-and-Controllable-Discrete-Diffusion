"""
Track the online class posterior p(c | x_t) while generating with the unconditional model,
and measure how well it predicts the V-gene family of the final sequence, taken as the
classification of real IgBLAST.

The image counterpart (../Image_experiments/scripts/posterior_classifier_agreement.py) is the
same diagnostic against a pretrained ViT instead of IgBLAST.

Generates sampling.posterior_num_samples sequences with guid_scale pinned at 0 (every
generative step driven by uncond_model only -- see sample_with_posterior's docstring in
src/discrete_flow_matching/sampling/sampling_standard.py), tracking a Bayesian class
posterior over all num_classes V-gene families at every step (num_classes+1 forward passes/
step: one batched cond_model call over every real class label + one uncond_model call).
Classifies the actually-decoded final sequences with real igblastp (compute_igblast_metrics),
then checks agreement between the posterior's final-step argmax ("what the model's own
conditional sub-networks believe they generated") and IgBLAST's verdict ("what was actually
generated"): an agreement rate + confusion matrix, plus (all purely post-hoc on the cached
per-step raw_logratios, no re-sampling) the full accuracy profile ported from
../Image_experiments/scripts/posterior_classifier_agreement.py --
  - agreement vs. diffusion time t
  - macro one-vs-rest AUC vs. diffusion time t (unnormalised evidence + a softmax-tau sweep)
  - temp sweep (NLL of the tracked posterior at the IgBLAST-called class, calibration only --
    argmax/agreement is provably temp-invariant)
  - macro one-vs-rest AUC and per-class ROC curves at the final step
  - predicted class distribution vs. the reference pool's own empirical V-gene-family frequency
  - agreement vs. tracked-posterior confidence margin (binned)

Model/dataset come from the Hydra compose (defaults list + overrides), NOT from a saved run
config. Example:
  python -m scripts.posterior_classifier_agreement \\
      'model=[amplify,35M,35M_vgene]' \\
      trainer=[base,ddlm] \\
      trainer.train.max_length=256 \\
      trainer.train.diffusion_type=uniform \\
      trainer.train.scheduler_type=linear \\
      sampling.run_name="oas95_igblast_human_35M_uniform" \\
      'sampling.to_evaluate_checkpoints=[20]' \\
      sampling.label_reference_csv=data/oas95_igblast_human/train.csv \\
      sampling.max_ref_seqs=20000 \\
      sampling.posterior_num_samples=256 \\
      sampling.num_steps=100 \\
      sampling.batch_size=32

Saves under:
  logs/Figures/{run_name}/posterior_unconditional/ckpt_{checkpoint}/n_{N}_steps_{S}_sched_{sched}/
    samples.csv              # sequence, tracked_pred, tracked_conf, tracked_margin,
                              #   igblast_family, igblast_pred_idx, agree,
                              #   post_prob_class_0..K-1 (final-step tracked posterior)
    post_probs.pt             # [T, N, K]  softmax posterior trajectory
    unnorm_post_probs.pt      # [T, N, K]  exp(accumulated log-likelihood) trajectory -- what
                               #   post_probs is a softmax OF, before normalisation across classes
    confusion.csv / confusion.png
    agreement_vs_time.csv / .png       # tracked-posterior-vs-IgBLAST agreement at each decile of t
    auc_vs_time_{tag}.csv / .png       # macro + per-class one-vs-rest AUC at every step, one
                                        #   variant per tag ("unnorm", "softmax_tau{T}" for each
                                        #   T in AUC_SOFTMAX_TAUS)
    temp_sweep.csv / .png              # NLL/agreement of the tracked posterior vs. tempering
    macro_auc.csv                      # final-step macro one-vs-rest AUC, "unnorm" + softmax_tau1.0
    roc_curves.csv / .png              # final-step one-vs-rest ROC curve per class (unnorm evidence)
    class_distribution.csv / .png      # tracked/IgBLAST class shares vs. the reference pool's
                                        #   own empirical V-gene-family frequency
    agreement_vs_margin.png            # agreement vs. tracked-posterior confidence margin, binned
    traj_{i}.png               # softmax posterior curve (sums to 1 at every t)
    traj_{i}_unnormalised.png  # raw exp(accumulated log-likelihood) curve (does NOT sum to 1 --
                               #   shows each class's evidence on its own, log-scaled y-axis)
"""
import os

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import auc, roc_auc_score, roc_curve

from scripts.sampling_utils import (
    VGENE_FAMILIES,
    build_classifier,
    build_x_in,
    compute_empirical_class_probs,
    compute_igblast_metrics,
    decode_clean,
    load_checkpoint,
)
from sequence_model.model.amplify_model import AMPLIFYConfig
from sequence_model.tokenizer import ProteinTokenizer
from scripts.plot_utils import (
    plot_agreement_vs_margin,
    plot_agreement_vs_time,
    plot_auc_vs_time,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_posterior_trajectory,
    plot_roc_curves,
    plot_temp_sweep,
)
from discrete_flow_matching.sampling import sample_with_posterior

SEED_BASE = 1  # reseed base -- see comment at the reseed call below
N_TRAJ_PLOTS = 8  # how many individual posterior-curve figures to save

# Constants for the AUC/ROC/calibration analyses below -- all purely post-hoc on the cached
# raw_logratios trajectory (ported from the image-experiment counterpart,
# ../Image_experiments/scripts/posterior_classifier_agreement.py), no re-sampling involved.
# Sampling is purely unconditional at temp=1.0 / offset=0.0 (sample_with_posterior's
# defaults), so the posterior columns in samples.csv already ARE the tau=1.0 softmax posterior.
# DEFAULT_TEMP is pinned to 1.0 to keep every "default" quantity here consistent with that.
# (The image counterpart sweeps guid_scale and uses 0.5 instead.)
DEFAULT_TEMP = 1.0
AUC_SOFTMAX_TAUS = [0.05, 0.1, 0.5, 1.0]  # tau values for the softmax AUC-vs-time curves
TEMP_SWEEP = torch.linspace(0.1, 2.0, 10)  # calibration sweep, centered around DEFAULT_TEMP=1.0
T_DECILES = [round(0.1 * i, 1) for i in range(11)]

# A class with too few IgBLAST-hit positive (or negative) examples gives an AUC estimate that's
# little better than a coin flip -- e.g. with only 1 positive example, that single sample's score
# can rank above or below the rest by chance alone, swinging the per-class AUC between ~0 and 1
# from one diffusion step to the next, and dragging the macro average along with it (see the rare
# V-gene families in the antibody dataset this was ported for: IGHV6/7 are ~0.5% of the reference
# pool, so a few hundred generated samples routinely yields 0-1 hits). Any class with fewer than
# this many hits on EITHER side (positive or negative) is excluded from every AUC/ROC computation
# below, the same way a class with literally zero hits already was.
MIN_CLASS_HITS_FOR_AUC = 10


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run_posterior_diagnostic(cfg: DictConfig):
    torch.manual_seed(SEED_BASE)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name                = cfg_dict["sampling"]["run_name"]
    uncond_run_name         = cfg_dict["sampling"].get("uncond_run_name", None)
    to_evaluate_checkpoints = cfg_dict["sampling"]["to_evaluate_checkpoints"]
    diffusion_type          = cfg_dict["trainer"]["train"]["diffusion_type"]
    scheduler_type          = cfg_dict["trainer"]["train"]["scheduler_type"]
    polynomial_power        = cfg_dict["trainer"]["train"]["polynomial_power"]
    max_length              = cfg_dict["trainer"]["train"]["max_length"]
    use_ancestral_sampling  = cfg_dict["sampling"]["use_ancestral_sampling"]
    num_steps  = cfg_dict["sampling"]["num_steps"]
    batch_size = cfg_dict["sampling"]["batch_size"]
    vocab_size = cfg_dict["tokenizer"]["vocab_size"]

    num_samples          = int(cfg_dict["sampling"]["posterior_num_samples"])
    label_reference_csv  = cfg_dict["sampling"]["label_reference_csv"]
    max_ref_seqs         = cfg_dict["sampling"].get("max_ref_seqs", None)
    germline_v           = cfg_dict["sampling"]["posterior_germline_v"]


    tokenizer = ProteinTokenizer(**cfg_dict["tokenizer"])
    model_config = AMPLIFYConfig(**cfg_dict["model"], **cfg_dict["tokenizer"])
    num_classes = model_config.num_classes
    if num_classes is None:
        raise ValueError(
            "model.num_classes is None -- pick a class-conditional model config group, "
            "e.g. 'model=[amplify,35M,35M_vgene]'."
        )
    uncond_idx = num_classes

    # classify_fn unused here on purpose -- ground truth comes from real IgBLAST below, not the
    # flank-identity heuristic. reference_df is only used to seed x_in from the whole pool.
    _, classifier_num_classes, class_names, reference_df = build_classifier(
        label_reference_csv, max_ref_seqs=max_ref_seqs,
    )
    if classifier_num_classes != num_classes:
        raise ValueError(
            f"The V-gene classifier has {classifier_num_classes} classes but the model "
            f"was built with num_classes={num_classes} -- these must match."
        )
    special_strs = {tokenizer.id_to_token(i) for i in tokenizer.special_token_ids}

    print(f"\nUnconditional posterior-tracking diagnostic "
          f"({num_classes}+1 = {num_classes + 1} forward passes/step)")
    print(f"  posterior_num_samples={num_samples}  num_steps={num_steps}  batch_size={batch_size}")
    if uncond_run_name is not None:
        print(f"  Using separately-trained unconditional model '{uncond_run_name}' for generation.")

    for checkpoint_number in to_evaluate_checkpoints:
        print(f"\n{'='*40}\nCheckpoint {checkpoint_number}\n{'='*40}")

        cond_model = load_checkpoint(run_name, checkpoint_number, model_config, device)
        uncond_model = cond_model
        if uncond_run_name is not None:
            uncond_model = load_checkpoint(uncond_run_name, checkpoint_number, model_config, device)

        out_dir = (f"logs/Figures/{run_name}/posterior_unconditional/ckpt_{checkpoint_number}/"
                   f"n_{num_samples}_steps_{num_steps}_sched_{scheduler_type}")
        os.makedirs(out_dir, exist_ok=True)

        # Reseed once (not per batch) so x_in is drawn consistently -- same convention as
        # sample_not_gate.py's shared x_in.
        seed = SEED_BASE
        torch.manual_seed(seed)
        x_in_shared = build_x_in(reference_df, tokenizer, max_length, num_samples, seed,
                                  diffusion_type, scheduler_type, polynomial_power, device,
                                  target_class=None)  # whole pool -- genuinely unconditional seeding

        c_uncond_full = torch.full((num_samples,), uncond_idx, dtype=torch.long, device=device)

        final_x_chunks, post_probs_chunks, unnorm_post_probs_chunks, raw_logratios_chunks = [], [], [], []
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            torch.manual_seed(seed + start)
            final_x_c, _, post_probs_c, unnorm_post_probs_c, raw_logratios_c = sample_with_posterior(
                cond_model, tokenizer, c_uncond_full[start:end], num_classes,
                x_in=x_in_shared[start:end].clone(),
                diffusion_type=diffusion_type, scheduler_type=scheduler_type, polynomial_power=polynomial_power,
                use_ancestral_sampling=use_ancestral_sampling,
                num_steps=num_steps, vocab_size=vocab_size,
                batch_size=end - start, length=max_length,
                return_trajectory=True,
                uncond_model=uncond_model,
            )
            final_x_chunks.append(final_x_c)
            post_probs_chunks.append(post_probs_c)
            unnorm_post_probs_chunks.append(unnorm_post_probs_c)
            raw_logratios_chunks.append(raw_logratios_c.cpu())
            print(f"  [{end}/{num_samples}] generated")

        final_x           = torch.cat(final_x_chunks, dim=0)
        # .cpu() here (not just at save time): every post-hoc analysis below indexes these with
        # plain CPU index/label tensors (scored_pos, igblast_label), so they need to already be
        # on CPU -- same convention as raw_logratios_chunks, appended .cpu() at accumulation time.
        post_probs        = torch.cat(post_probs_chunks, dim=1).cpu()         # [T, N, K]
        unnorm_post_probs = torch.cat(unnorm_post_probs_chunks, dim=1).cpu()  # [T, N, K]
        torch.save(post_probs, os.path.join(out_dir, "post_probs.pt"))
        torch.save(unnorm_post_probs, os.path.join(out_dir, "unnorm_post_probs.pt"))

        # raw_logratios [T, N, K] is temp/offset-independent (sample_with_posterior's
        # docstring): cache its cumulative sum so every AUC/ROC/temp-sweep analysis below can be
        # recomputed for any tau/temp value purely post-hoc, without re-running the model. Sampling
        # itself used temp=1.0/offset=0.0 (the function's defaults), so cum_logratios_t reconstructs
        # post_probs/unnorm_post_probs exactly at tau=1.0.
        raw_logratios = torch.cat(raw_logratios_chunks, dim=1)     # [T, N, K]
        cum_logratios_t = torch.cumsum(raw_logratios, dim=0)       # [T, N, K]

        seqs = [decode_clean(tokenizer, special_strs, final_x[i]) for i in range(num_samples)]
        tracked_probs = post_probs[-1].cpu()          # [N, K]  final-step softmax posterior
        tracked_pred  = tracked_probs.argmax(dim=-1)   # [N]
        tracked_conf  = tracked_probs.max(dim=-1).values
        tracked_top2   = torch.topk(tracked_probs, k=2, dim=-1).values  # [N, 2], sorted descending
        tracked_margin = tracked_top2[:, 0] - tracked_top2[:, 1]        # top1 - top2 confidence gap

        print(f"\nClassifying {num_samples} generated sequences with real IgBLAST...")
        ids = [f"gen_{i}" for i in range(num_samples)]
        igblast_result = compute_igblast_metrics(seqs, ids, germline_v, tmp_prefix=f"{out_dir}/igblast_tmp")
        per_seq = igblast_result["per_sequence"]  # same row order as seqs/ids, built by construction
        igblast_family = per_seq["igblast_family"]
        igblast_pred_idx = igblast_family.map(
            lambda f: VGENE_FAMILIES.index(f) if f in VGENE_FAMILIES else None
        )

        df = pd.DataFrame({
            "sample_number": range(num_samples),
            "sequence": seqs,
            "tracked_pred": tracked_pred.tolist(),
            "tracked_conf": tracked_conf.tolist(),
            "tracked_margin": tracked_margin.tolist(),
            "igblast_family": igblast_family.values,
            "igblast_pred_idx": igblast_pred_idx.values,
            "igblast_no_hit": per_seq["no_hit"].values,
        })
        for k in range(num_classes):
            df[f"post_prob_class_{k}"] = tracked_probs[:, k].tolist()

        match = df["tracked_pred"] == df["igblast_pred_idx"]
        df["agree"] = match.where(df["igblast_pred_idx"].notna())

        scored = df[df["igblast_pred_idx"].notna()].copy()
        scored["igblast_pred_idx"] = scored["igblast_pred_idx"].astype(int)

        n_hit = len(scored)
        agree_rate = scored["agree"].mean() if n_hit else float("nan")
        print(f"\n--- Checkpoint {checkpoint_number} summary (N={num_samples}) ---")
        print(f"  IgBLAST hit rate: {n_hit}/{num_samples} ({n_hit / num_samples:.3f})")
        print(f"  Tracked-posterior vs IgBLAST agreement (of hits): {agree_rate:.4f}"
              f"  (chance level = {1.0 / num_classes:.4f})")

        csv_path = os.path.join(out_dir, "samples.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Saved per-sample results to {csv_path}")

        # Everything below is scored purely post-hoc against IgBLAST-hit samples only (a
        # no-hit sequence has no ground-truth label to score against), using the SAME cached
        # cum_logratios_t / post_probs -- no re-sampling for any of it.
        scored_pos       = torch.from_numpy(scored.index.values)
        igblast_label    = torch.from_numpy(scored["igblast_pred_idx"].to_numpy(dtype=np.int64))
        cum_logratios_t_scored = cum_logratios_t[:, scored_pos, :]   # [T, N_hit, K]
        cum_logratios_scored   = cum_logratios_t_scored[-1]          # [N_hit, K]  (== final step)

        # --- Agreement vs. diffusion time t ---
        # Label: 1 iff IgBLAST's call on the FINAL sequence matches the tracked posterior's
        # argmax at time t (same reference as agree_rate above, evaluated at every decile) --
        # this is what exposes a late flip (agreement rising, then dropping back near t=1).
        t_next_values = torch.linspace(0, 1, num_steps)[1:]
        t_target_indices = [int(torch.argmin((t_next_values - tt).abs())) for tt in T_DECILES]
        t_target_actual  = [round(t_next_values[idx].item(), 4) for idx in t_target_indices]

        agreement_vs_time = []
        for idx in t_target_indices:
            pred_t = post_probs[idx, scored_pos, :].argmax(dim=-1).cpu()
            agreement_vs_time.append((pred_t == igblast_label).float().mean().item())

        print(f"\n  --- Agreement vs. diffusion time t (tracked posterior vs IgBLAST) ---")
        for tt, ta, ag in zip(T_DECILES, t_target_actual, agreement_vs_time):
            print(f"    t={tt:.1f} (actual t_next={ta:.3f}): agreement={ag:.4f}")

        time_df = pd.DataFrame({
            "t_target": T_DECILES, "t_actual": t_target_actual,
            "agreement": agreement_vs_time, "n": n_hit,
        })
        time_df.to_csv(os.path.join(out_dir, "agreement_vs_time.csv"), index=False)

        plot_agreement_vs_time(
            T_DECILES, agreement_vs_time, num_classes,
            f"Agreement vs. t -- ckpt {checkpoint_number}, N={n_hit}",
            os.path.join(out_dir, "agreement_vs_time.png"))
        print(f"  Saved agreement-vs-time results/plot to {out_dir}/agreement_vs_time.{{csv,png}}")

        # --- Macro one-vs-rest AUC vs. diffusion time t, at EVERY sampler step ---
        # Label: 1 iff IgBLAST classifies the FINAL sequence as c. AUC_c(t) is "how well does
        # the evidence for c at time t discern the samples that end up as c from the rest";
        # averaged over classes. Two score families, each read off for a single class c and
        # ranked across samples:
        #   'unnorm'  : cum_logratios_t[:, :, c] -- the unnormalised class-vs-uncond log-odds
        #               (no softmax). Cross-sample AUC needs this to be comparable across
        #               samples; ROC/AUC are invariant to any monotonic rescaling of the score,
        #               so no tau choice is needed for this one.
        #   'softmax' : p(c|x_t) = softmax(tau * cum_logratios_t)[:, :, c] at each tau in
        #               AUC_SOFTMAX_TAUS -- the K-way posterior mass on c. Unlike 'unnorm',
        #               this genuinely depends on tau (softmax is a nonlinear function of the
        #               full K-vector), so several tau values are swept rather than fixed to one.
        # An explicit t=0 point is prepended: no evidence has accumulated yet, every score is 0
        # (all tied; softmax is uniform 1/K), so AUC = 0.5 exactly by definition. A class with
        # fewer than MIN_CLASS_HITS_FOR_AUC positives or negatives among the IgBLAST-hit samples
        # is skipped (too unstable to trust, see MIN_CLASS_HITS_FOR_AUC's docstring) and excluded
        # from that time point's average.
        auc_t_axis = np.concatenate([[0.0], t_next_values.numpy()])  # [T+1]
        valid_classes = [c for c in range(num_classes)
                          if MIN_CLASS_HITS_FOR_AUC <= (igblast_label == c).sum().item()
                          <= len(igblast_label) - MIN_CLASS_HITS_FOR_AUC]

        auc_time_variants = {"unnorm": cum_logratios_t_scored}
        for tau in AUC_SOFTMAX_TAUS:
            auc_time_variants[f"softmax_tau{tau}"] = torch.softmax(tau * cum_logratios_t_scored, dim=-1)

        for tag, score_tnk in auc_time_variants.items():
            auc_class_arr = np.full((len(auc_t_axis), num_classes), np.nan)
            auc_class_arr[0, valid_classes] = 0.5
            for ti in range(score_tnk.shape[0]):
                scores_t = score_tnk[ti].numpy()  # [N_hit, K]
                for c in valid_classes:
                    auc_class_arr[ti + 1, c] = roc_auc_score((igblast_label.numpy() == c).astype(int), scores_t[:, c])
            auc_macro = np.nanmean(auc_class_arr, axis=1)

            auc_time_df = pd.DataFrame({
                "t_target": auc_t_axis, "auc": auc_macro,
                "n_classes_used": len(valid_classes), "n": n_hit,
            })
            for c in range(num_classes):
                auc_time_df[f"auc_class_{c}"] = auc_class_arr[:, c]
            auc_time_df.to_csv(os.path.join(out_dir, f"auc_vs_time_{tag}.csv"), index=False)
            print(f"  Macro one-vs-rest AUC vs t [{tag}]: t=0 -> {auc_macro[0]:.4f}, t=0.5 -> "
                  f"{auc_macro[int(np.argmin(np.abs(auc_t_axis - 0.5)))]:.4f}, t=1 -> {auc_macro[-1]:.4f} "
                  f"({len(valid_classes)}/{num_classes} classes)")

            plot_auc_vs_time(
                auc_t_axis, auc_class_arr, auc_macro, valid_classes, tag,
                f"AUC vs. t [{tag}] -- ckpt {checkpoint_number}, N={n_hit}",
                os.path.join(out_dir, f"auc_vs_time_{tag}.png"))
        print(f"  Saved AUC-vs-time results/plots ({len(auc_time_variants)} variants) to {out_dir}/auc_vs_time_*")

        # --- Temp sweep: NLL of the tracked posterior at the IgBLAST-called class ---
        # Reuses cum_logratios_scored -- no re-sampling needed, since the posterior is linear
        # in temp (sample_with_posterior's docstring). Agreement rate is also
        # reported for documentation, but it is provably temp-invariant (positive scalar
        # multiplication never changes argmax), so it is expected flat.
        n_hit_t = cum_logratios_scored.shape[0]
        sweep_records = []
        for temp_val in TEMP_SWEEP.tolist():
            probs = torch.softmax(temp_val * cum_logratios_scored, dim=-1)
            pred  = probs.argmax(dim=-1)
            agreement_t = (pred == igblast_label).float().mean().item()
            nll_t = -torch.log(probs[torch.arange(n_hit_t), igblast_label] + 1e-12).mean().item()
            sweep_records.append({"temp": temp_val, "agreement": agreement_t, "nll": nll_t})

        sweep_df = pd.DataFrame(sweep_records)
        best_temp = sweep_df.loc[sweep_df["nll"].idxmin(), "temp"]
        print(f"  Temp sweep: NLL ranges [{sweep_df['nll'].min():.4f}, {sweep_df['nll'].max():.4f}], "
              f"best temp={best_temp:.3f} (sampling itself used temp={DEFAULT_TEMP}); agreement is flat "
              f"at {sweep_df['agreement'].mean():.4f} (std={sweep_df['agreement'].std():.2e}, as expected).")
        sweep_df.to_csv(os.path.join(out_dir, "temp_sweep.csv"), index=False)

        plot_temp_sweep(
            sweep_df["temp"], sweep_df["nll"], sweep_df["agreement"], best_temp,
            f"Temp sweep -- ckpt {checkpoint_number}, N={n_hit_t}",
            os.path.join(out_dir, "temp_sweep.png"))
        print(f"  Saved temp-sweep results/plot to {out_dir}/temp_sweep.{{csv,png}}")

        # --- Macro-averaged one-vs-rest AUC, over ALL classes, final step only ---
        # Same scores as the ROC section below, but averaged across every class instead of
        # cherry-picking one, for an honest summary number. A class is skipped (and excluded
        # from the average) if it has fewer than MIN_CLASS_HITS_FOR_AUC positive or negative
        # examples in the IgBLAST-hit population (too unstable to trust -- same threshold as
        # valid_classes above, re-applied here since this section computes its own AUCs from
        # scratch rather than reusing valid_classes/auc_class_arr).
        macro_auc_records = []
        macro_auc_score_fns = {
            "unnorm": lambda c: cum_logratios_scored[:, c].numpy(),
            f"softmax_tau{DEFAULT_TEMP}": lambda c: torch.softmax(DEFAULT_TEMP * cum_logratios_scored, dim=-1)[:, c].numpy(),
        }
        for score_name, score_fn in macro_auc_score_fns.items():
            per_class_auc = {}
            for c in range(num_classes):
                label = (igblast_label == c).numpy().astype(int)
                n_pos = int(label.sum())
                if n_pos < MIN_CLASS_HITS_FOR_AUC or (len(label) - n_pos) < MIN_CLASS_HITS_FOR_AUC:
                    continue
                fpr, tpr, _ = roc_curve(label, score_fn(c))
                per_class_auc[c] = auc(fpr, tpr)
            macro_auc = float(np.mean(list(per_class_auc.values()))) if per_class_auc else float("nan")
            print(f"  Macro one-vs-rest AUC [{score_name}] over {len(per_class_auc)}/{num_classes} classes: {macro_auc:.4f}")
            for c, class_auc in per_class_auc.items():
                macro_auc_records.append({"score": score_name, "class": class_names[c], "auc": class_auc})
            macro_auc_records.append({"score": score_name, "class": "macro_avg", "auc": macro_auc})

        macro_auc_df = pd.DataFrame(macro_auc_records)
        macro_auc_df.to_csv(os.path.join(out_dir, "macro_auc.csv"), index=False)
        print(f"  Saved macro one-vs-rest AUC results to {out_dir}/macro_auc.csv")

        # --- ROC curves: one-vs-rest per class, does the model's own internal (unnormalised)
        # evidence for class c predict what IgBLAST eventually calls the (unconditionally-
        # generated) sequence? Derived from the SAME cached cum_logratios_scored -- purely
        # post-hoc, no re-sampling. Unlike the AUC-vs-time softmax variants above, ROC/AUC are
        # invariant to any monotonic rescaling of the score, so the temp-invariant 'unnorm'
        # evidence alone is enough here (no tau choice to sweep).
        if not valid_classes:
            print(f"  Skipping ROC curves: no class has both positive and negative IgBLAST-hit "
                  f"examples among N={n_hit_t} samples.")
        else:
            roc_curves_to_plot, roc_records = [], []
            for c in valid_classes:
                label = (igblast_label == c).numpy().astype(int)
                score = cum_logratios_scored[:, c].numpy()
                fpr, tpr, thresholds = roc_curve(label, score)
                class_auc = auc(fpr, tpr)
                print(f"  ROC [unnorm] for class {class_names[c]} vs IgBLAST: AUC={class_auc:.4f} "
                      f"(n_pos={int(label.sum())}/{n_hit_t})")
                # color=None: each class takes the next colour from matplotlib's cycle.
                roc_curves_to_plot.append((fpr, tpr, f"{class_names[c]} (AUC={class_auc:.3f})", None))
                for f, t, th in zip(fpr, tpr, thresholds):
                    roc_records.append({"class": class_names[c], "fpr": f, "tpr": t, "threshold": th})

            pd.DataFrame(roc_records).to_csv(os.path.join(out_dir, "roc_curves.csv"), index=False)
            plot_roc_curves(
                roc_curves_to_plot,
                f"ROC -- one-vs-rest per class, unnormalised evidence\n"
                f"ckpt {checkpoint_number}, N={n_hit_t}",
                os.path.join(out_dir, "roc_curves.png"))
            print(f"  Saved ROC curve results/plot to {out_dir}/roc_curves.{{csv,png}}")

        # --- Confusion matrix: tracked posterior vs IgBLAST ---
        confusion = np.zeros((num_classes, num_classes), dtype=int)
        for _, row in scored.iterrows():
            confusion[int(row["igblast_pred_idx"]), int(row["tracked_pred"])] += 1

        plot_confusion_matrix(
            confusion, class_names,
            f"Confusion -- ckpt {checkpoint_number}, agreement={agree_rate:.3f}, N={n_hit}",
            os.path.join(out_dir, "confusion.png"))
        pd.DataFrame(confusion, index=class_names, columns=class_names).to_csv(
            os.path.join(out_dir, "confusion.csv"))
        print(f"  Saved confusion matrix to {out_dir}/confusion.png")

        # --- Predicted class distribution: how often each class comes out, tracked posterior
        # vs IgBLAST, against the reference pool's OWN empirical class frequency. Unlike the
        # balanced CIFAR/MNIST classes in the images script (uniform 1/K is the natural
        # reference line there), real V-gene-family frequencies are naturally imbalanced (e.g.
        # IGHV3 dominant) -- compute_empirical_class_probs gives the honest reference instead of
        # assuming uniformity. Uniform 1/K is also drawn for context.
        tracked_counts = scored["tracked_pred"].value_counts().reindex(range(num_classes), fill_value=0)
        igblast_counts = scored["igblast_pred_idx"].value_counts().reindex(range(num_classes), fill_value=0)
        tracked_pct = 100.0 * tracked_counts / n_hit
        igblast_pct = 100.0 * igblast_counts / n_hit
        empirical_pct = 100.0 * compute_empirical_class_probs(reference_df, num_classes)

        plot_class_distribution(
            tracked_pct.values, igblast_pct.values, empirical_pct, class_names,
            f"Predicted class distribution -- ckpt {checkpoint_number}, N={n_hit}",
            os.path.join(out_dir, "class_distribution.png"))
        pd.DataFrame({
            "class": class_names, "tracked_pct": tracked_pct.values, "igblast_pct": igblast_pct.values,
            "empirical_pct": empirical_pct,
        }).to_csv(os.path.join(out_dir, "class_distribution.csv"), index=False)
        print(f"  Saved class-distribution results/plot to {out_dir}/class_distribution.{{csv,png}}")

        # --- A handful of per-trajectory posterior-curve plots ---
        t_axis = torch.linspace(0, 1, post_probs.shape[0])
        n_plot = min(N_TRAJ_PLOTS, num_samples)
        for i in range(n_plot):
            igb_idx = df.loc[i, "igblast_pred_idx"]
            igb_idx = int(igb_idx) if pd.notna(igb_idx) else None
            igb_name = class_names[igb_idx] if igb_idx is not None else "no hit"
            tracked_name = class_names[int(tracked_pred[i])]
            plot_posterior_trajectory(
                t_axis, post_probs[:, i, :].cpu(), class_names,
                f"Tracked posterior (final: {tracked_name})  vs  IgBLAST: {igb_name}",
                r"$p(c|x_t)$  (softmax, $\sum_c p(c|x_t) = 1$)",
                os.path.join(out_dir, f"traj_{i}.png"))
            plot_posterior_trajectory(
                t_axis, unnorm_post_probs[:, i, :].cpu(), class_names,
                f"Unnormalised tracked posterior (final: {tracked_name})  vs  IgBLAST: {igb_name}",
                r"$\exp(\mathrm{accumulated\ log\text{-}likelihood})$  (not normalised)",
                os.path.join(out_dir, f"traj_{i}_unnormalised.png"), log_scale=True)
        print(f"  Saved {n_plot} trajectory plots (normalised + unnormalised) to {out_dir}/")

        # --- Agreement vs. tracked-posterior confidence margin (top1 - top2), binned in 0.1 steps ---
        bin_edges = np.linspace(0.0, 1.0, 11)  # 0-0.1, 0.1-0.2, ..., 0.9-1.0
        df["margin_bin"] = pd.cut(df["tracked_margin"], bins=bin_edges, include_lowest=True)
        binned = df.groupby("margin_bin", observed=False)["agree"].agg(["mean", "count"])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        valid_bins = binned["count"] > 0
        plot_agreement_vs_margin(
            bin_centers[valid_bins], binned["mean"][valid_bins], binned["count"][valid_bins],
            num_classes,
            f"Agreement vs. confidence margin -- ckpt {checkpoint_number}, N={n_hit}",
            os.path.join(out_dir, "agreement_vs_margin.png"))
        print(f"  Saved agreement-vs-margin plot to {out_dir}/agreement_vs_margin.png")


if __name__ == "__main__":
    run_posterior_diagnostic()
