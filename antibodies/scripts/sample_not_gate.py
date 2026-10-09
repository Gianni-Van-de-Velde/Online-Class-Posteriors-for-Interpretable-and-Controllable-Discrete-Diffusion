"""
Generate sequences that avoid one V-gene family, with the NOT gate or with either baseline.
`sampling.not_method` picks which:

  "dynamic"        the NOT gate (sample_NOT): the negative guidance scale is driven by the
                   online class posterior. Sweeps all combinations of sampling.not_temps,
                   sampling.not_initial_guid_scales and sampling.not_offsets for one forbidden
                   class, and saves the posterior / guidance-scale trajectories alongside the
                   sequences.
  "unconditional"  sample() at guid_scale=0, run ONCE per checkpoint. guid_scale=0 reduces to
                   the unconditional model whatever is in `c`, so generation is
                   class-independent: one pool is scored against every forbidden class
                   downstream rather than regenerated per class.
  "static"         sample() at a constant guid_scale<0 per forbidden class, swept over
                   sampling.not_static_guid_scales (magnitudes -- the sign is applied here).
                   Classifier-free guidance pushed away from the class at every denoising step,
                   against the dynamic method's per-step scale.

All three start from the same x_in pool (SEED_BASE), so differences between them are not down
to different random draws. Add sampling.not_use_prior_weighted_init=true to the dynamic method
to read the swept initial_guid_scale as the value a class would get under a BALANCED prior:
each sample then starts at r*p(c)/(1-p(c)) with r = initial_guid_scale*(num_classes-1) and
p(c) the empirical class frequency of label_reference_csv (see sample_NOT's docstring). Those
runs get a '_priorinit' folder suffix so the two modes never collide on disk.

Writes, per checkpoint, one folder per setting:
  logs/Figures/{run_name}/not_gate[_{tag}]/ckpt_{k}/
    not_{class}_temp_{T:.4f}_initgs_{IGS:.4f}_offset_{OFF:.4f}_gsmax_{GSMAX:.4f}
    _steps_{steps}_sched_{sched}_n_{N}[_priorinit]/
      samples.csv      sample_number, sequence, predicted_class, is_violation
      post_probs.pt    [T, N] unnormalised p_c(x_t) trajectory
      guid_scales.pt   [T, N] realised (post-clamp) guid_scale trajectory
  logs/Figures/{run_name}/not_gate_baseline[_{tag}]/ckpt_{k}/
    unconditional_steps_{steps}_sched_{sched}_n_{N}/
      samples.csv      sample_number, sequence, predicted_class  (class-independent, so no
                       is_violation column)
    static_{class}_gs_{S:.2f}_steps_{steps}_sched_{sched}_n_{N}/
      samples.csv      sample_number, sequence, predicted_class, is_violation

Read back by metrics_not_gate.py. Example (the invocation run/gates.sh uses):
  python -m scripts.sample_not_gate \\
      'model=[amplify,35M,35M_vgene]' \\
      trainer=[base,ddlm] \\
      trainer.train.max_length=256 \\
      sampling.run_name="oas95_igblast_human_35M_uniform" \\
      'sampling.to_evaluate_checkpoints=[20]' \\
      sampling.label_reference_csv=data/oas95_igblast_human/train.csv \\
      sampling.not_method=dynamic \\
      sampling.not_class=0 \\
      sampling.not_n_samples=4096 \\
      'sampling.not_temps=[0.01]' \\
      'sampling.not_initial_guid_scales=[0.1,0.5,2.0]' \\
      'sampling.not_offsets=[0.0]'
"""
import os

import hydra
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
from discrete_flow_matching.sampling import sample, sample_NOT

METHODS = ("dynamic", "unconditional", "static")

# Batch b of every setting is seeded SEED_BASE + b, in all three methods, so they share a
# starting x_in pool and the same random substrate.
SEED_BASE = 1


def tagged_samples_root(run_name, base, tag=""):
    """Sample-output root, optionally suffixed by a run tag ("" = the untagged layout).

    A tag sends a fresh sweep to a sibling directory. Use one whenever not_n_samples changes:
    the metrics scripts dedup on hyperparameters and not on sample count, so mixing sample
    counts in one tree silently gives a CSV whose rows are not comparable.
    """
    return f"logs/Figures/{run_name}/{base}" + (f"_{tag}" if tag else "")


def build_sweep(method, cfg, num_classes, class_priors, scheduler_type):
    """One entry per folder this run will produce: (folder name, forbidden class, the sampler
    arguments that distinguish it). Everything else about the call is shared across methods.

    A forbidden class of None means the pool is class-independent (the unconditional baseline),
    which is also what decides whether samples.csv gets an is_violation column.
    """
    n, steps = cfg["not_n_samples"], cfg["num_steps"]
    tail = f"_steps_{steps}_sched_{scheduler_type}_n_{n}"

    if method == "unconditional":
        return [(f"unconditional{tail}", None, dict(guid_scale=0.0))]

    if method == "static":
        classes = cfg.get("not_static_classes") or list(range(num_classes))
        return [(f"static_{c}_gs_{-s:.2f}{tail}", c, dict(guid_scale=-s))
                for c in classes for s in cfg["not_static_guid_scales"]]

    not_class = cfg["not_class"]
    gsmax = cfg["not_guid_scale_max"]
    prior_init = bool(cfg.get("not_use_prior_weighted_init", False))
    sweep = []
    for temp in cfg["not_temps"]:
        for igs in cfg["not_initial_guid_scales"]:
            for offset in cfg["not_offsets"]:
                # .4f, not .2f: the swept temps/offsets go down to 0.001, and at 2 decimals
                # 0.001 and 0.005 both render as "0.00". Colliding names silently drop grid
                # points -- the second folder is skipped as "already exists", and the first is
                # then read back by metrics_not_gate.py under the WRONG temp/offset, since it
                # parses both out of this name.
                # '_priorinit' only when the flag is on: the same (temp, igs, offset) means a
                # different start value in the two modes, so they must not collide on disk.
                fname = (f"not_{not_class}_temp_{temp:.4f}_initgs_{igs:.4f}_offset_{offset:.4f}"
                          f"_gsmax_{gsmax:.4f}{tail}"
                          f"{'_priorinit' if prior_init else ''}")
                sweep.append((fname, not_class, dict(
                    guid_scale_max=gsmax, initial_guid_scale=igs,
                    use_prior_weighted_init=prior_init, class_priors=class_priors,
                    temp=temp, offset=offset,
                )))
    return sweep


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
    max_length              = cfg_dict["trainer"]["train"]["max_length"]
    use_ancestral_sampling  = cfg_dict["sampling"]["use_ancestral_sampling"]
    num_steps  = cfg_dict["sampling"]["num_steps"]
    vocab_size = cfg_dict["tokenizer"]["vocab_size"]

    method              = cfg_dict["sampling"]["not_method"]
    n_samples           = int(cfg_dict["sampling"]["not_n_samples"])
    sample_batch_size   = int(cfg_dict["sampling"].get("not_sample_batch_size") or n_samples)
    sample_batch_size   = min(sample_batch_size, n_samples)
    not_tag             = str(cfg_dict["sampling"].get("not_tag", "") or "")
    label_reference_csv = cfg_dict["sampling"]["label_reference_csv"]
    max_ref_seqs        = cfg_dict["sampling"].get("max_ref_seqs", None)
    if method not in METHODS:
        raise ValueError(f"sampling.not_method must be one of {METHODS}, got {method!r}")

    tokenizer = ProteinTokenizer(**cfg_dict["tokenizer"])
    model_config = AMPLIFYConfig(**cfg_dict["model"], **cfg_dict["tokenizer"])
    num_classes = model_config.num_classes
    if num_classes is None:
        raise ValueError(
            "model.num_classes is None -- pick a class-conditional model config group, "
            "i.e. 'model=[amplify,35M,35M_vgene]'."
        )
    uncond_idx = num_classes

    classify_fn, classifier_num_classes, class_names, reference_df = build_classifier(
        label_reference_csv, max_ref_seqs=max_ref_seqs,
    )
    if classifier_num_classes != num_classes:
        raise ValueError(
            f"The V-gene classifier has {classifier_num_classes} classes but the model "
            f"was built with num_classes={num_classes} -- these must match."
        )
    special_strs = {tokenizer.id_to_token(i) for i in tokenizer.special_token_ids}

    # Empirical p(c) over the same reference pool the classifier was built from, so the
    # prior-weighted start value is consistent with what the metrics are scored against.
    class_priors = None
    if method == "dynamic" and cfg_dict["sampling"].get("not_use_prior_weighted_init", False):
        class_priors = compute_empirical_class_probs(reference_df, num_classes)
        prior_str = "  ".join(f"{class_names[k]}={class_priors[k]:.4f}" for k in range(num_classes))
        print(f"  Prior-weighted init ON -- empirical p(c): {prior_str}")

    sweep = build_sweep(method, cfg_dict["sampling"], num_classes, class_priors, scheduler_type)
    print(f"\nNOT-gate sampler: method={method}, {len(sweep)} setting(s), "
          f"{n_samples} sequences each")
    if uncond_run_name is not None:
        print(f"  Using separately-trained unconditional model '{uncond_run_name}'.")
    print()

    base_dir = "not_gate" if method == "dynamic" else "not_gate_baseline"
    out_base_root = tagged_samples_root(run_name, base_dir, not_tag)

    for checkpoint_number in to_evaluate_checkpoints:
        print(f"\n{'='*40}\nCheckpoint {checkpoint_number}\n{'='*40}")

        cond_model = load_checkpoint(run_name, checkpoint_number, model_config, device)
        uncond_model = cond_model
        if uncond_run_name is not None:
            uncond_model = load_checkpoint(uncond_run_name, checkpoint_number, model_config, device)

        out_base = f"{out_base_root}/ckpt_{checkpoint_number}"
        os.makedirs(out_base, exist_ok=True)

        # Reseed once (not per setting) so every setting starts from the exact same x_in and
        # consumes the same underlying random substrate -- common random numbers across the
        # sweep, and across the three methods, so metric differences reflect the swept
        # hyperparameter rather than which random reference sequences got drawn.
        seed = SEED_BASE
        torch.manual_seed(seed)
        x_in_shared = build_x_in(reference_df, tokenizer, max_length, n_samples, seed,
                                  diffusion_type, scheduler_type, polynomial_power, device,
                                  target_class=None)  # full pool -- the gate steers away from a
                                                       # class, it does not target a length
        c_uncond = torch.full((n_samples,), uncond_idx, dtype=torch.long, device=device)

        for fname, not_class, extra in sweep:
            save_dir = os.path.join(out_base, fname)
            csv_path = os.path.join(save_dir, "samples.csv")
            if os.path.exists(csv_path):
                print(f"  Skipping {fname} (already exists)")
                continue
            os.makedirs(save_dir, exist_ok=True)

            # not_class is None for the unconditional pool, where guid_scale=0 makes the label
            # a no-op whatever it is.
            c = torch.full((n_samples,), 0 if not_class is None else not_class,
                           dtype=torch.long, device=device)

            # Chunked over sample_batch_size so n_samples can exceed what fits in one
            # forward-pass batch. Each chunk is reseeded to seed+start, so chunks get
            # independent randomness while the SAME chunk index gets the same seed in every
            # setting -- which is what preserves the common-random-numbers comparison. With
            # not_sample_batch_size >= n_samples this is a single chunk seeded with `seed`.
            final_x_chunks, post_probs_chunks, guid_scales_chunks = [], [], []
            for start in range(0, n_samples, sample_batch_size):
                end = min(start + sample_batch_size, n_samples)
                torch.manual_seed(seed + start)
                shared = dict(
                    x_in=x_in_shared[start:end].clone(),
                    diffusion_type=diffusion_type, scheduler_type=scheduler_type,
                    polynomial_power=polynomial_power,
                    use_ancestral_sampling=use_ancestral_sampling,
                    num_steps=num_steps, vocab_size=vocab_size,
                    batch_size=end - start, length=max_length,
                    uncond_model=uncond_model,
                )
                if method == "dynamic":
                    final_x_c, _, post_probs_c, guid_scales_c = sample_NOT(
                        cond_model, tokenizer, c[start:end], c_uncond[start:end],
                        return_trajectory=True, **shared, **extra)
                    post_probs_chunks.append(post_probs_c)
                    guid_scales_chunks.append(guid_scales_c)
                else:
                    final_x_c, _ = sample(
                        cond_model, tokenizer, c[start:end], c_uncond[start:end],
                        return_trajectory=False, **shared, **extra)
                final_x_chunks.append(final_x_c)

            final_x = torch.cat(final_x_chunks, dim=0)
            if method == "dynamic":
                torch.save(torch.cat(post_probs_chunks, dim=1).cpu(),
                           os.path.join(save_dir, "post_probs.pt"))
                torch.save(torch.cat(guid_scales_chunks, dim=1).cpu(),
                           os.path.join(save_dir, "guid_scales.pt"))

            rows = []
            for i in range(n_samples):
                seq = decode_clean(tokenizer, special_strs, final_x[i])
                pred = classify_fn(seq)
                row = {"sample_number": i, "sequence": seq, "predicted_class": pred}
                if not_class is not None:
                    row["is_violation"] = (pred == not_class)
                rows.append(row)
            pd.DataFrame(rows).to_csv(csv_path, index=False)

            if not_class is None:
                print(f"  {fname}: {n_samples} sequences -> {csv_path}")
            else:
                n_viol = sum(r["is_violation"] for r in rows)
                print(f"  {fname}: {n_viol}/{n_samples} classified as forbidden class "
                      f"{not_class} -> {csv_path}")


if __name__ == "__main__":
    main()
