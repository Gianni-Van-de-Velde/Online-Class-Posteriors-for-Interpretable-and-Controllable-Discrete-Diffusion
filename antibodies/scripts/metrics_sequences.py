"""
Metrics for the class-conditional samples written by sample_sequences.py: naturalness,
distributional metrics and real-IgBLAST confidence, all computed in one pass per sample folder.

Naturalness: IgLM log-likelihood (Shuai et al., 2021). IgLM is trained autoregressively on
OAS antibody chains conditioned on chain/species tokens, so unlike a general-protein AR model
its score reads as "how antibody-like is this" rather than "how protein-like". Reported for
three groups per folder:
  - generated   the model's own samples (from that folder's samples.csv)
  - reference   real training sequences of the same target class (cached per class -- doesn't
                depend on num_steps/guid_scale, so it's computed once and reused)
  - random      uniform-random amino acids, length-matched to the generated sequences
                (floor: what "not natural at all" scores like)

Distributional: FPD / precision / recall / C2ST against the target class's reference
sequences, embedded with ESM2 or IgBert (see --use_igbert).

IgBLAST confidence: real `igblastp` -- not the flank-identity heuristic used at sampling time
-- against the same IMGT human IGHV germline db that produced the training labels: hit_rate,
weak_hit_rate, mean %identity/coverage of the top hit, and match_rate against the folder's
target class.

Reads:  logs/Figures/{run_name}/conditional/ckpt_{k}/<one folder per setting>/samples.csv
Writes: logs/Results/{run_name}/conditional_results.csv, one row per setting
        (settings already in the CSV are skipped unless --force)

Example (the invocation run/sample_sweep.sh uses):
  python -m scripts.metrics_sequences \\
      --run_name oas95_igblast_human_35M_uniform \\
      --to_evaluate_checkpoints 20 \\
      --label_reference_csv data/oas95_igblast_human/train.csv \\
      --max_ref_seqs 20000
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from scripts.sampling_utils import (
    append_result_row,
    build_classifier,
    compute_igblast_metrics,
    model_tag,
    parse_folder,
    reference_pool_tag,
    slugify,
    sort_folders_by_setting,
)
from discrete_flow_matching.metrics import (
    ESM2Detector,
    IgBertDetector,
    IgLMEvaluator,
    get_embeddings_in_batches,
    sequence_metrics,
)

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def generate_random_sequences(lengths, seed=0):
    """Uniform-random amino-acid sequences, one per length in `lengths` -- the
    length-matched floor baseline (no antibody structure at all, but a fair
    comparison since IgLM log-likelihood is length-dependent)."""
    rng = np.random.default_rng(seed)
    return [
        "".join(rng.choice(AMINO_ACIDS, size=length))
        for length in lengths
    ]


def score_iglm(evaluator, seqs, chain_token, species_token, desc="IgLM"):
    """Per-sequence IgLM log-likelihoods.

    A sequence IgLM's tokenizer can't score gets NaN instead of crashing the whole sweep --
    IgLM's vocab is only the 20 standard amino acids + its own special tokens (no 'B'/'X'/etc.
    ambiguity codes, no AMPLIFY's '|' chain-separator token), so a near-random checkpoint
    (e.g. checkpoint 0, sampled uniformly over AMPLIFY's full 27-token vocab) reliably emits
    residues IgLM's `log_likelihood` rejects via an internal AssertionError. Use nan_mean_std
    below rather than .mean()/.std() directly on the result."""
    lls = torch.full((len(seqs),), float("nan"))
    n_failed = 0
    for i, seq in enumerate(tqdm(seqs, desc=f"  {desc}", leave=False)):
        try:
            lls[i] = evaluator.get_iglm_log_likelihood(seq, chain_token, species_token)
        except (AssertionError, ValueError, IndexError):
            n_failed += 1
    if n_failed:
        print(f"    [{desc}] {n_failed}/{len(seqs)} sequences unscoreable by IgLM "
              f"(non-standard residues) -- recorded as NaN")
    return lls


def nan_mean_std(lls):
    """Mean/std over the non-NaN entries of a score_iglm() result. Returns (nan, nan) if
    every sequence in the batch was unscoreable."""
    valid = lls[~torch.isnan(lls)]
    if len(valid) == 0:
        return float("nan"), float("nan")
    return valid.mean().item(), valid.std().item()


def get_ref_feats(detector, reference_df, target_class, cache_dir, batch_size=64):
    """Embeddings of the reference sequences of one class (cached).

    target_class="natural" -- the natural-frequency sampling mode -- uses the WHOLE reference
    pool instead of one class's subset, since the generated batch itself mixes classes in
    their real-world proportions."""
    cache_path = os.path.join(cache_dir, f"embeddings_class_{target_class}.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu", weights_only=True)
    if target_class == "natural":
        seqs = reference_df["sequence"].astype(str).tolist()
    else:
        seqs = reference_df[reference_df["label"] == target_class]["sequence"].astype(str).tolist()
    feats = get_embeddings_in_batches(detector, seqs, batch_size=batch_size).cpu()
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(feats, cache_path)
    return feats


def get_class_ref_lls(evaluator, reference_df, target_class, cache_dir, chain_token, species_token,
                       num_ref_samples, seed):
    """Reference-sequence IgLM log-likelihoods for one class -- independent of num_steps/
    guid_scale, so cached once and reused across every folder for that class/checkpoint.

    target_class="natural" (the natural-frequency sampling mode) uses the WHOLE (unfiltered)
    reference pool instead of one class's subset, matching that mode's generated batch, which
    itself mixes classes in their real-world proportions."""
    cache_path = os.path.join(cache_dir, f"ref_lls_class_{target_class}_n{num_ref_samples}.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    if target_class == "natural":
        pool = reference_df["sequence"].astype(str).tolist()
    else:
        pool = reference_df[reference_df["label"] == target_class]["sequence"].astype(str).tolist()
    rng = np.random.default_rng(seed)
    seqs = [pool[i] for i in rng.choice(len(pool), size=min(num_ref_samples, len(pool)), replace=False)]

    lls = score_iglm(evaluator, seqs, chain_token, species_token, desc="IgLM-ref")
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(lls, cache_path)
    return lls


def compute_metrics(folder_path, config, iglm_evaluator, detector, reference_df,
                     iglm_cache_dir, embedding_cache_dir, chain_token, species_token,
                     model_name, num_ref_samples, seed, class_names=None, germline_v=None):
    csv_path = os.path.join(folder_path, "samples.csv")
    if not os.path.exists(csv_path):
        print(f"    no samples.csv in {folder_path}, skipping")
        return None
    df = pd.read_csv(csv_path)
    gen_seqs = df["sequence"].astype(str).tolist()

    gen_lls = score_iglm(iglm_evaluator, gen_seqs, chain_token, species_token, desc="IgLM-gen")
    ref_lls = get_class_ref_lls(iglm_evaluator, reference_df, config["class"], iglm_cache_dir,
                                 chain_token, species_token, num_ref_samples, seed)

    random_seqs = generate_random_sequences([len(s) for s in gen_seqs], seed=seed)
    random_lls = score_iglm(iglm_evaluator, random_seqs, chain_token, species_token, desc="IgLM-random")

    gen_feats = get_embeddings_in_batches(detector, gen_seqs, batch_size=64).cpu()
    ref_feats = get_ref_feats(detector, reference_df, config["class"], embedding_cache_dir)
    dist_metrics = sequence_metrics(ref_feats, gen_feats)

    gen_ll_mean, gen_ll_std = nan_mean_std(gen_lls)
    ref_ll_mean, ref_ll_std = nan_mean_std(ref_lls)
    random_ll_mean, random_ll_std = nan_mean_std(random_lls)

    row = {
        # str(), not the raw value: config["class"] is an int for a fixed-class folder but the
        # string "natural" for the natural-frequency mode, and `class` is a dedup key. A CSV
        # column holding both comes back from read_csv as object dtype -- i.e. EVERY value is a
        # str, including the ints -- so an int key built on a later run would no longer match
        # the row it wrote, and that run would re-score the whole sweep and then append
        # duplicates instead of replacing. Stringifying at the only place the value is
        # persisted keeps the column single-typed in both directions. This is byte-identical on
        # disk (CSV carries no types), so the committed results files stay valid as they are.
        # Note it is only the ROW that is stringified: config["class"] itself stays an int
        # below and above, because the reference-pool lookups index `label` with it.
        "class":          str(config["class"]),
        "guid_scale":     config["guid_scale"],
        "num_steps":      config["num_steps"],
        "scheduler_type": config["scheduler_type"],
        "n_generated":    len(gen_seqs),
        "chain_token":    chain_token,
        "species_token":  species_token,
        "iglm_model":     model_name,
        "gen_ll_mean":    gen_ll_mean,
        "gen_ll_std":     gen_ll_std,
        "ref_ll_mean":    ref_ll_mean,
        "ref_ll_std":     ref_ll_std,
        "random_ll_mean": random_ll_mean,
        "random_ll_std":  random_ll_std,
        "gen_ppl":        float(np.exp(-gen_ll_mean)),
        "ref_ppl":        float(np.exp(-ref_ll_mean)),
        "random_ppl":     float(np.exp(-random_ll_mean)),
        "fpd":            dist_metrics["fpd"],
        "precision":      dist_metrics["precision"],
        "recall":         dist_metrics["recall"],
        "c2st_accuracy":  dist_metrics["c2st_accuracy"],
        "c2st_log_loss":  dist_metrics["c2st_log_loss"],
    }

    if germline_v is not None:
        if config["class"] == "natural":
            # Natural-frequency mode: class varies per row (samples.csv's own `class` column,
            # written by sample_natural_frequency), not one fixed folder-level target class.
            requested_families = [
                class_names[c] if class_names is not None and 0 <= c < len(class_names) else None
                for c in df["class"].astype(int).tolist()
            ]
        else:
            requested_family = (class_names[config["class"]]
                                 if class_names is not None and config["class"] < len(class_names) else None)
            requested_families = [requested_family] * len(gen_seqs)
        gen_ids = [f"s{i}" for i in range(len(gen_seqs))]
        tmp_prefix = "/tmp/igblast_" + re.sub(r"[^A-Za-z0-9_.-]", "_", folder_path)
        igblast_metrics = compute_igblast_metrics(
            gen_seqs, gen_ids, germline_v,
            requested_families=requested_families, tmp_prefix=tmp_prefix,
        )
        row.update({
            "igblast_hit_rate":      igblast_metrics["igblast_hit_rate"],
            "igblast_weak_hit_rate": igblast_metrics["igblast_weak_hit_rate"],
            "igblast_mean_pident":   igblast_metrics["igblast_mean_pident"],
            "igblast_mean_coverage": igblast_metrics["igblast_mean_coverage"],
            "igblast_match_rate":    igblast_metrics["igblast_match_rate"],
        })

    return row


def main():
    parser = argparse.ArgumentParser(
        description="One-stop metrics sweep: IgLM naturalness, fpd/precision/recall/C2ST, and real-IgBLAST confidence/match")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--to_evaluate_checkpoints", nargs="+", type=int, required=True)
    parser.add_argument("--label_reference_csv", default="data/oas95_igblast_human/train.csv")
    parser.add_argument("--target_class", type=int, default=None,
                         help="Filter to folders with this target class. If omitted, all classes are evaluated.")
    parser.add_argument("--folder_name", default=None, help="Evaluate a single specific folder instead of discovering all.")
    parser.add_argument("--num_ref_samples", type=int, default=256, help="Reference sequences sampled per class for the IgLM ll baseline.")
    parser.add_argument("--max_ref_seqs", type=int, default=20000,
                         help="Load a uniform-random N-row sample of --label_reference_csv instead of the full file; "
                              "0 loads it in full. The reference pool is what FPD/C2ST are measured against, so "
                              "changing this gives numbers not comparable to the shipped CSVs (which used 20000).")
    parser.add_argument("--chain_token", type=str, default="[HEAVY]", help="IgLM chain token")
    parser.add_argument("--species_token", type=str, default="[HUMAN]", help="IgLM species token")
    parser.add_argument("--model", type=str, default="IgLM", choices=["IgLM", "IgLM-S"])
    parser.add_argument("--use_igbert", action="store_true",
                         help="Embed with antibody-specific IgBert (~420M, slow) instead of ESM2 (~35M, fast).")
    parser.add_argument("--embedding_model_name", default=None,
                         help="Override the embedding model checkpoint. Defaults to facebook/esm2_t12_35M_UR50D, "
                              "or Exscientia/IgBert if --use_igbert is set.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the random-sequence baseline and reference subsampling.")
    parser.add_argument("--germline_v", default="data/igblast_germline/human/human_IGHV_aa",
                         help="IMGT V-gene germline db passed to igblastp -query -germline_db_V.")
    parser.add_argument("--skip_igblast", action="store_true",
                         help="Skip the real-igblastp confidence/match metrics.")
    parser.add_argument("--force", action="store_true", help="Recompute metrics even for settings already present in the results CSV.")
    args = parser.parse_args()
    if args.embedding_model_name is None:
        args.embedding_model_name = "Exscientia/IgBert" if args.use_igbert else "facebook/esm2_t12_35M_UR50D"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = args.run_name

    do_igblast = not args.skip_igblast

    print(f"\nRun: {run_name}")
    print(f"Checkpoints: {args.to_evaluate_checkpoints}")
    print(f"Chain/species: {args.chain_token} / {args.species_token}")
    print(f"IgBLAST confidence/match: {'on' if do_igblast else 'off'}")

    _, _, class_names, reference_df = build_classifier(
        args.label_reference_csv, max_ref_seqs=args.max_ref_seqs or None, seed=args.seed,
    )

    print(f"Loading {args.model} evaluator...")
    iglm_evaluator = IgLMEvaluator(model_name=args.model)
    if args.use_igbert:
        print("Loading IgBert detector...")
        detector = IgBertDetector(model_name=args.embedding_model_name, device=device)
    else:
        print("Loading ESM2 detector...")
        detector = ESM2Detector(model_name=args.embedding_model_name, device=device)

    dedup_keys = ["run_name", "checkpoint", "class", "guid_scale", "num_steps", "scheduler_type"]
    csv_path = f"logs/Results/{run_name}/conditional_results.csv"

    existing_keys = set()
    if not args.force and os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        # `class` is read back as int64 if the file happens to hold only fixed-class rows, and
        # as str if any "natural" row is present -- so normalise it here rather than trusting
        # whichever read_csv inferred. See append_result_row's INVARIANT note.
        existing_df["class"] = existing_df["class"].astype(str)
        existing_keys = set(existing_df[dedup_keys].itertuples(index=False, name=None))
        print(f"Found {len(existing_keys)} settings already in {csv_path} -- these will be skipped (use --force to recompute).")

    for checkpoint_number in args.to_evaluate_checkpoints:
        samples_dir = f"logs/Figures/{run_name}/conditional/ckpt_{checkpoint_number}"
        if not os.path.isdir(samples_dir):
            print(f"\nCheckpoint {checkpoint_number}: no samples dir at {samples_dir}, skipping.")
            continue

        print(f"\n{'='*50}\nCheckpoint {checkpoint_number}\n{'='*50}")

        if args.folder_name:
            folders = [args.folder_name]
        else:
            folders = sort_folders_by_setting([
                f for f in os.listdir(samples_dir)
                if os.path.isdir(os.path.join(samples_dir, f)) and parse_folder(f) is not None
            ])
        if args.target_class is not None:
            folders = [f for f in folders if (c := parse_folder(f)) is not None and c["class"] == args.target_class]

        print(f"Found {len(folders)} folders to evaluate")
        # Both caches are keyed by everything their contents depend on: the reference pool
        # (csv + max_ref_seqs + seed) and the model that produced the tensors. Keying only on
        # the class -- as these did -- means a second run with a different --max_ref_seqs or a
        # different --embedding_model_name silently reads the first run's tensors and reports
        # FPD / C2ST / the IgLM reference baseline against a pool it never loaded. See
        # reference_pool_tag.
        pool = reference_pool_tag(args.label_reference_csv, args.max_ref_seqs or None, args.seed)
        iglm_cache_dir = os.path.join(
            "data/iglm_ref_cache",
            # chain/species tokens too: they condition IgLM, so they change the scores.
            f"{slugify(args.model)}_{slugify(args.chain_token)}_{slugify(args.species_token)}",
            pool)
        embedding_cache_dir = os.path.join(
            "data/embeddings_cache", model_tag(args.embedding_model_name), pool)
        print(f"Reference caches: {embedding_cache_dir}  |  {iglm_cache_dir}")

        for folder in folders:
            config = parse_folder(folder)
            if config is None:
                continue
            # str(config["class"]) to match how the row is written -- see the note on `class`
            # in compute_metrics. Everything else is single-typed and compares as itself.
            key = (run_name, checkpoint_number, str(config["class"]), config["guid_scale"],
                   config["num_steps"], config["scheduler_type"])
            if key in existing_keys:
                print(f"  Skipping {folder} -- already in {csv_path} (use --force to recompute)")
                continue

            print(f"\n── {folder}")
            row = compute_metrics(
                folder_path=os.path.join(samples_dir, folder),
                config=config,
                iglm_evaluator=iglm_evaluator,
                detector=detector,
                reference_df=reference_df,
                iglm_cache_dir=iglm_cache_dir,
                embedding_cache_dir=embedding_cache_dir,
                chain_token=args.chain_token,
                species_token=args.species_token,
                model_name=args.model,
                num_ref_samples=args.num_ref_samples,
                seed=args.seed,
                class_names=class_names,
                germline_v=(args.germline_v if do_igblast else None),
            )
            if row is None:
                continue
            row["run_name"] = run_name
            row["checkpoint"] = checkpoint_number
            row["embedding_model"] = args.embedding_model_name
            # The reference pool FPD / C2ST / precision / recall were measured against. Not a
            # dedup key -- re-scoring a setting should replace its row -- but recorded, because
            # without it a CSV can hold rows scored against different pools and look uniform,
            # and a re-run that disagrees with the shipped numbers gives no way to see why.
            row["max_ref_seqs"] = args.max_ref_seqs or None
            row["ref_seed"] = args.seed
            existing_keys.add(key)

            append_result_row(csv_path, row, dedup_keys, str_dedup_keys=("class",))

            igblast_summary = (
                f"  igblast_match={row['igblast_match_rate']:.3f} (hit_rate={row['igblast_hit_rate']:.3f})"
                if "igblast_match_rate" in row else ""
            )
            print(f"    gen_ppl={row['gen_ppl']:.3f}  ref_ppl={row['ref_ppl']:.3f}  random_ppl={row['random_ppl']:.3f}  "
                  f"fpd={row['fpd']:.2f}  P={row['precision']:.3f}  R={row['recall']:.3f}  "
                  f"C2ST_acc={row['c2st_accuracy']:.3f}{igblast_summary}  [saved to {csv_path}]")

    print(f"\nResults saved incrementally to {csv_path}")
    if not os.path.exists(csv_path):
        print("No results computed.")
        return

    full_df = pd.read_csv(csv_path)
    full_df = full_df[full_df["checkpoint"].isin(args.to_evaluate_checkpoints)]
    full_df = full_df.sort_values(["guid_scale", "class", "scheduler_type", "num_steps"])
    summary_cols = ["checkpoint", "class", "guid_scale", "num_steps", "gen_ppl", "ref_ppl", "random_ppl",
                     "fpd", "precision", "recall", "c2st_accuracy",
                     "igblast_hit_rate", "igblast_match_rate"]
    summary_cols = [c for c in summary_cols if c in full_df.columns]
    print(full_df[summary_cols].to_string(index=False))


if __name__ == "__main__":
    main()
