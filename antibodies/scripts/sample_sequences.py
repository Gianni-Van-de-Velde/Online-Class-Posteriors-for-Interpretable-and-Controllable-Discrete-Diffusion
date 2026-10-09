"""
Generate class-conditional samples with classifier-free guidance, sweeping guid_scale
(0=unconditional, 1=plain conditional, >1=CFG-amplified) and num_steps.

sampling.classes picks the mode:
  given (e.g. [0,1,...,6])  per-class sweep: n_per_class sequences for EACH listed class,
                            one folder per (class, guid_scale, num_steps)
  null (the default)        natural-frequency mode: n_per_class sequences TOTAL, with the
                            class labels drawn i.i.d. from the reference pool's own empirical
                            frequency, so a rare family like IGHV7 stays rare instead of
                            being 1/K. One folder, with a `class` column varying per row.

Every setting starts from the same x_in for a given class (common random numbers), so metric
differences across the sweep reflect the swept hyperparameter rather than the random draws.

Reads:  logs/<run>/checkpoints/checkpoint_<k>/model.safetensors
        sampling.label_reference_csv  (real sequences, to seed x_in and to draw labels from)
Writes: logs/Figures/<run>/conditional/ckpt_<k>/class_<c|natural>_gs_<g>_steps_<n>_sched_<s>_n_<N>/
          samples.csv   # sample_number, sequence, class, guid_scale, num_steps
Read back by metrics_sequences.py.

The generated sequences are NOT classified here: that is a brute-force scan against the whole
reference pool, so it lives in scripts/classify_vgene_samples.py as an opt-in step.

Example (the invocation run/sample_sweep.sh uses):
  python -m scripts.sample_sequences \\
      'model=[amplify,35M,35M_vgene]' \\
      trainer=[base,ddlm] \\
      trainer.train.max_length=256 \\
      sampling.run_name="oas95_igblast_human_35M_uniform" \\
      'sampling.to_evaluate_checkpoints=[20]' \\
      sampling.n_per_class=4096 \\
      sampling.n_per_class_num_batches=32 \\
      'sampling.guid_scales=[0.0,1.0]' \\
      'sampling.num_steps_list=[4,8,16,32,64,128,256]'
"""
import os

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from scripts.sampling_utils import (
    build_classifier,
    build_x_in,
    compute_empirical_class_probs,
    decode_clean,
    load_checkpoint,
)
from sequence_model.model.amplify_model import AMPLIFYConfig
from sequence_model.tokenizer import ProteinTokenizer
from discrete_flow_matching.sampling import sample

SEED_BASE = 1  # per-class reseed base -- see comment at the reseed call below


def batch_bounds(n_total, num_batches):
    """Even split of n_total into num_batches (start, end) ranges, remainder spread over the
    first few. num_batches=1 -> [(0, n_total)], i.e. one unchunked batch."""
    base, remainder = divmod(n_total, num_batches)
    bounds, pos = [], 0
    for b in range(num_batches):
        size = base + (1 if b < remainder else 0)
        bounds.append((pos, pos + size))
        pos += size
    return bounds


def _run_setting(cond_model, uncond_model, tokenizer, c, c_uncond, x_in, bounds, seed,
                  diffusion_type, scheduler_type, polynomial_power, use_ancestral_sampling,
                  vocab_size, max_length, steps, gs):
    """One (class-assignment, guid_scale, num_steps) setting, chunked into len(bounds)
    sequential forward-pass batches (each with its own seed) and concatenated."""
    final_x_chunks = []
    for start, end in bounds:
        torch.manual_seed(seed + start)
        final_x_b, _ = sample(
            cond_model, tokenizer, c[start:end], c_uncond[start:end],
            guid_scale=gs, x_in=x_in[start:end].clone(),
            diffusion_type=diffusion_type, scheduler_type=scheduler_type, polynomial_power=polynomial_power,
            use_ancestral_sampling=use_ancestral_sampling,
            num_steps=steps, vocab_size=vocab_size,
            batch_size=end - start, length=max_length,
            return_trajectory=False,
            uncond_model=uncond_model,
        )
        final_x_chunks.append(final_x_b)
    return torch.cat(final_x_chunks, dim=0)


def sample_class_sweep(cond_model, uncond_model, tokenizer, reference_df, classes, uncond_idx,
                        n_per_class, num_batches, guid_scales, num_steps_list, scheduler_type,
                        diffusion_type, polynomial_power, use_ancestral_sampling, vocab_size,
                        max_length, device, special_strs, out_base):
    """n_per_class sequences for EACH class in `classes`, one folder per
    (class, guid_scale, num_steps). Used whenever sampling.classes is given."""
    for cls in classes:
        # Reseed per class, not per setting, so every (guid_scale, num_steps) combination
        # for this class starts from the same x_in -- common random numbers across the sweep.
        seed = SEED_BASE + cls
        torch.manual_seed(seed)
        x_in_cls = build_x_in(reference_df, tokenizer, max_length, n_per_class, seed,
                               diffusion_type, scheduler_type, polynomial_power, device,
                               target_class=cls)

        c        = torch.full((n_per_class,), cls,        dtype=torch.long, device=device)
        c_uncond = torch.full((n_per_class,), uncond_idx, dtype=torch.long, device=device)
        bounds   = batch_bounds(n_per_class, num_batches)

        for steps in num_steps_list:
            for gs in guid_scales:
                fname = f"class_{cls}_gs_{gs:.2f}_steps_{steps}_sched_{scheduler_type}_n_{n_per_class}"
                save_dir = os.path.join(out_base, fname)
                csv_path = os.path.join(save_dir, "samples.csv")
                if os.path.exists(csv_path):
                    print(f"  Skipping {fname} (already exists)")
                    continue
                os.makedirs(save_dir, exist_ok=True)

                final_x = _run_setting(cond_model, uncond_model, tokenizer, c, c_uncond, x_in_cls,
                                        bounds, seed, diffusion_type, scheduler_type, polynomial_power,
                                        use_ancestral_sampling, vocab_size, max_length, steps, gs)

                rows = []
                for i in range(n_per_class):
                    seq = decode_clean(tokenizer, special_strs, final_x[i])
                    rows.append({
                        "sample_number": i,
                        "sequence": seq,
                        "class": cls,
                        "guid_scale": gs,
                        "num_steps": steps,
                    })
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                print(f"  {fname}: {n_per_class} sequences ({len(bounds)} batch(es)) -> {csv_path}")


def sample_natural_frequency(cond_model, uncond_model, tokenizer, reference_df, num_classes, uncond_idx,
                              n_total, num_batches, guid_scales, num_steps_list, scheduler_type,
                              diffusion_type, polynomial_power, use_ancestral_sampling, vocab_size,
                              max_length, device, special_strs, out_base):
    """n_total class labels drawn once, i.i.d. from the reference pool's empirical frequency,
    and shared across the whole sweep (common random numbers, as above).

    x_in comes from the FULL reference pool rather than being filtered per drawn label, so the
    starting length is not matched to the class a sample happens to be assigned -- by design,
    since the point of this mode is to reproduce the natural class mixture."""
    seed = SEED_BASE
    probs = compute_empirical_class_probs(reference_df, num_classes)
    rng = np.random.default_rng(seed)
    drawn_labels = rng.choice(num_classes, size=n_total, p=probs)
    counts = dict(zip(*np.unique(drawn_labels, return_counts=True)))
    print(f"  Empirical class probs: {[round(p, 4) for p in probs]}")
    print(f"  Drawn label counts (n={n_total}): {counts}")

    torch.manual_seed(seed)
    x_in = build_x_in(reference_df, tokenizer, max_length, n_total, seed,
                       diffusion_type, scheduler_type, polynomial_power, device)

    c        = torch.tensor(drawn_labels, dtype=torch.long, device=device)
    c_uncond = torch.full((n_total,), uncond_idx, dtype=torch.long, device=device)
    bounds   = batch_bounds(n_total, num_batches)

    for steps in num_steps_list:
        for gs in guid_scales:
            fname = f"class_natural_gs_{gs:.2f}_steps_{steps}_sched_{scheduler_type}_n_{n_total}"
            save_dir = os.path.join(out_base, fname)
            csv_path = os.path.join(save_dir, "samples.csv")
            if os.path.exists(csv_path):
                print(f"  Skipping {fname} (already exists)")
                continue
            os.makedirs(save_dir, exist_ok=True)

            final_x = _run_setting(cond_model, uncond_model, tokenizer, c, c_uncond, x_in,
                                    bounds, seed, diffusion_type, scheduler_type, polynomial_power,
                                    use_ancestral_sampling, vocab_size, max_length, steps, gs)

            rows = []
            for i in range(n_total):
                seq = decode_clean(tokenizer, special_strs, final_x[i])
                rows.append({
                    "sample_number": i,
                    "sequence": seq,
                    "class": int(drawn_labels[i]),
                    "guid_scale": gs,
                    "num_steps": steps,
                })
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"  {fname}: {n_total} sequences ({len(bounds)} batch(es)) -> {csv_path}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    torch.manual_seed(SEED_BASE)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name                = cfg_dict["sampling"]["run_name"]
    uncond_run_name         = cfg_dict["sampling"].get("uncond_run_name", None)
    to_evaluate_checkpoints = cfg_dict["sampling"]["to_evaluate_checkpoints"]
    diffusion_type          = cfg_dict["trainer"]["train"]["diffusion_type"]
    scheduler_type          = cfg_dict["trainer"]["train"]["scheduler_type"]
    polynomial_power        = cfg_dict["trainer"]["train"]["polynomial_power"]
    max_length               = cfg_dict["trainer"]["train"]["max_length"]
    use_ancestral_sampling    = cfg_dict["sampling"]["use_ancestral_sampling"]
    vocab_size = cfg_dict["tokenizer"]["vocab_size"]

    n_per_class          = cfg_dict["sampling"]["n_per_class"]
    num_batches          = int(cfg_dict["sampling"].get("n_per_class_num_batches", 1))
    guid_scales          = [float(v) for v in cfg_dict["sampling"]["guid_scales"]]
    num_steps_list       = [int(v) for v in cfg_dict["sampling"]["num_steps_list"]]
    label_reference_csv  = cfg_dict["sampling"]["label_reference_csv"]
    max_ref_seqs         = cfg_dict["sampling"].get("max_ref_seqs", None)

    tokenizer = ProteinTokenizer(**cfg_dict["tokenizer"])
    model_config = AMPLIFYConfig(**cfg_dict["model"], **cfg_dict["tokenizer"])
    num_classes = model_config.num_classes
    if num_classes is None:
        raise ValueError(
            "model.num_classes is None -- pick a class-conditional model config group, "
            "i.e. 'model=[amplify,35M,35M_vgene]'."
        )
    uncond_idx = num_classes

    # classify_fn is deliberately unused: see the note in the module docstring.
    _classify_fn, classifier_num_classes, class_names, reference_df = build_classifier(
        label_reference_csv, max_ref_seqs=max_ref_seqs,
    )
    if classifier_num_classes != num_classes:
        raise ValueError(
            f"The V-gene classifier has {classifier_num_classes} classes but the model "
            f"was built with num_classes={num_classes} -- these must match."
        )

    classes_cfg = cfg_dict["sampling"].get("classes", None)
    natural_mode = not classes_cfg   # None or [] -> draw labels from the reference pool's own frequency
    classes = classes_cfg if classes_cfg else list(range(num_classes))
    special_strs = {tokenizer.id_to_token(i) for i in tokenizer.special_token_ids}

    if natural_mode:
        print(f"\nNatural-frequency conditional sweep (no sampling.classes given): "
              f"guid_scales={guid_scales}  num_steps={num_steps_list}")
        print(f"  {n_per_class} sequences total, class labels drawn from the reference pool's "
              f"own frequency (pass sampling.classes=[...] to sweep specific classes instead)"
              + (f", split into {num_batches} batches" if num_batches > 1 else ""))
    else:
        print(f"\nConditional CFG sweep: classes={classes}  guid_scales={guid_scales}  num_steps={num_steps_list}")
        print(f"  {n_per_class} sequences per (class, guid_scale, num_steps) combo"
              + (f", split into {num_batches} batches" if num_batches > 1 else ""))
    print(f"  {num_classes} V-gene classes: {class_names}")
    if uncond_run_name is not None:
        print(f"  Using separately-trained unconditional model '{uncond_run_name}' for every uncond-conditioned forward pass.")
    print()

    out_base_root = f"logs/Figures/{run_name}/conditional"

    for checkpoint_number in to_evaluate_checkpoints:
        print(f"\n{'='*40}\nCheckpoint {checkpoint_number}\n{'='*40}")

        cond_model = load_checkpoint(run_name, checkpoint_number, model_config, device)
        uncond_model = cond_model
        if uncond_run_name is not None:
            uncond_model = load_checkpoint(uncond_run_name, checkpoint_number, model_config, device)

        out_base = f"{out_base_root}/ckpt_{checkpoint_number}"
        os.makedirs(out_base, exist_ok=True)

        if natural_mode:
            sample_natural_frequency(
                cond_model, uncond_model, tokenizer, reference_df, num_classes, uncond_idx,
                n_per_class, num_batches, guid_scales, num_steps_list, scheduler_type,
                diffusion_type, polynomial_power, use_ancestral_sampling, vocab_size,
                max_length, device, special_strs, out_base,
            )
        else:
            sample_class_sweep(
                cond_model, uncond_model, tokenizer, reference_df, classes, uncond_idx,
                n_per_class, num_batches, guid_scales, num_steps_list, scheduler_type,
                diffusion_type, polynomial_power, use_ancestral_sampling, vocab_size,
                max_length, device, special_strs, out_base,
            )


if __name__ == "__main__":
    main()
