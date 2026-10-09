"""
Score the NOT-gate samples written by sample_not_gate.py -- the dynamic gate and both
baselines -- so all three methods' numbers come from the same machinery and are directly
comparable.

Per folder:
  violation_rate       fraction of generated sequences classified AS the forbidden class
                       (the gate's failure metric)
  n_parsed             how many sequences the classifier could parse at all
  fpd                  Frechet Protein Distance against the "allowed classes" reference subset
  precision, recall    k-NN precision/recall against the same subset (--skip_precision_recall)
  iglm_gen_ll_mean/std IgLM log-likelihood of the generated sequences -- naturalness, the other
                       half of the accuracy/quality tradeoff (--skip_iglm)

Only violation_rate and fpd depend on the forbidden class. The unconditional baseline is ONE
class-independent pool, so it is scored once against each class in turn, with the embedding and
IgLM passes -- by far the expensive part -- done once for the pool rather than once per class.
The other two methods produce one pool per class, which is the same loop with one iteration.

  default    reads logs/Figures/{run}/not_gate[_{tag}]/,          writes not_gate_results.csv
  --baseline reads logs/Figures/{run}/not_gate_baseline[_{tag}]/, writes
             not_gate_baseline_results.csv

Two CSVs, because the methods are described by different hyperparameters: temp / initial scale
/ offset / init rule for the dynamic gate, a single constant guid_scale for the baselines.

Example (the invocations run/gates.sh uses):
  python -m scripts.metrics_not_gate \\
      --run_name oas95_igblast_human_35M_uniform --to_evaluate_checkpoints 20 --tag final \\
      --label_reference_csv data/oas95_igblast_human/train.csv \\
      --max_ref_seqs 20000 --skip_precision_recall
  python -m scripts.metrics_not_gate --baseline --skip_reuse_unconditional ...   # same flags
"""
import argparse
import glob
import os
import re

import pandas as pd
import torch

from scripts.metrics_sequences import nan_mean_std, score_iglm
from scripts.sample_not_gate import tagged_samples_root
from scripts.sampling_utils import (
    append_result_row,
    build_classifier,
    model_tag,
    reference_pool_tag,
)
from discrete_flow_matching.metrics import (
    ESM2Detector,
    IgLMEvaluator,
    calculate_frechet_distance,
    get_embeddings_in_batches,
    get_embeddings_statistics,
    knn_precision_recall_features,
)

# The trailing '_priorinit' marks folders sampled with
# sampling.not_use_prior_weighted_init=true, where initial_guid_scale is a ratio against the
# prior odds rather than the flat start value (see sample_NOT's docstring). The same
# (temp, igs, offset) means a different run in the two modes, so it is carried through as its
# own column and dedup key.
DYNAMIC_RE = re.compile(
    r"^not_(?P<not_class>\d+)_temp_(?P<temp>[0-9.]+)_initgs_(?P<initial_guid_scale>[0-9.]+)"
    r"_offset_(?P<offset>-?[0-9.]+)_gsmax_(?P<guid_scale_max>[0-9.]+)"
    r"_steps_(?P<num_steps>\d+)_sched_(?P<scheduler_type>\w+)_n_(?P<n_samples>\d+)"
    r"(?P<priorinit>_priorinit)?$"
)
UNCOND_RE = re.compile(r"^unconditional_steps_(\d+)_sched_(\w+)_n_(\d+)$")
STATIC_RE = re.compile(r"^static_(\d+)_gs_(-?[0-9.]+)_steps_(\d+)_sched_(\w+)_n_(\d+)$")

# Which hyperparameters identify a row, per results CSV. This is the only thing that differs
# between the two beyond which columns describe the method.
DEDUP_KEYS = {
    "not_gate_results.csv": ["run_name", "checkpoint", "not_class", "temp", "initial_guid_scale",
                              "offset", "guid_scale_max", "num_steps", "scheduler_type",
                              "use_prior_weighted_init"],
    "not_gate_baseline_results.csv": ["run_name", "checkpoint", "method", "not_class",
                                       "guid_scale", "num_steps", "scheduler_type"],
}


def parse_folder(name):
    """Sampling settings from a folder name, or None if it is not a NOT-gate sample folder."""
    m = DYNAMIC_RE.match(name)
    if m:
        g = m.groupdict()
        return {
            "method":             "dynamic",
            "not_class":          int(g["not_class"]),
            "temp":               float(g["temp"]),
            "initial_guid_scale": float(g["initial_guid_scale"]),
            "offset":             float(g["offset"]),
            "guid_scale_max":     float(g["guid_scale_max"]),
            "num_steps":          int(g["num_steps"]),
            "scheduler_type":     g["scheduler_type"],
            "n_samples":          int(g["n_samples"]),
            "use_prior_weighted_init": g["priorinit"] is not None,
        }
    m = UNCOND_RE.match(name)
    if m:
        return {"method": "unconditional", "not_class": None, "guid_scale": 0.0,
                "num_steps": int(m.group(1)), "scheduler_type": m.group(2),
                "n_samples": int(m.group(3))}
    m = STATIC_RE.match(name)
    if m:
        return {"method": "static", "not_class": int(m.group(1)), "guid_scale": float(m.group(2)),
                "num_steps": int(m.group(3)), "scheduler_type": m.group(4),
                "n_samples": int(m.group(5))}
    return None


def tagged_results_path(run_name, filename, tag="", shard=None):
    """Where a metrics run writes its results CSV, optionally tagged and sharded.

    tag="" gives logs/Results/{run_name}/{filename}. A tag nests the run in its own
    subdirectory so a fresh sweep never appends to a CSV holding incomparable rows.

    shard is what makes parallel scoring safe: each shard gets its own file under shards/,
    merged afterwards by scripts/merge_metric_shards.py. See append_result_row for why
    concurrent jobs cannot share one path.
    """
    results_dir = f"logs/Results/{run_name}" + (f"/{tag}" if tag else "")
    if shard is not None:
        results_dir = os.path.join(results_dir, "shards")
        stem, ext = os.path.splitext(filename)
        filename = f"{stem}.shard{shard}{ext}"
    return os.path.join(results_dir, filename)


_REF_FEATS_CACHE = {}
_REF_STATS_CACHE = {}


def get_allowed_ref_feats(detector, reference_df, not_class, cache_dir, batch_size=64):
    """Embeddings of the reference sequences of every class EXCEPT not_class -- what a run that
    successfully avoids that class should look like.

    Memoised in process on top of the on-disk cache: the tensor is ~77 MB and the disk cache
    usually lives on NFS, so re-loading it once per folder is pure overhead across a sweep.
    """
    if not_class in _REF_FEATS_CACHE:
        return _REF_FEATS_CACHE[not_class]
    cache_path = os.path.join(cache_dir, f"esm_embeddings_allowed_not_{not_class}.pt")
    if os.path.exists(cache_path):
        feats = torch.load(cache_path, map_location="cpu", weights_only=True)
    else:
        seqs = reference_df[reference_df["label"] != not_class]["sequence"].astype(str).tolist()
        feats = get_embeddings_in_batches(detector, seqs, batch_size=batch_size).cpu()
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(feats, cache_path)
    _REF_FEATS_CACHE[not_class] = feats
    return feats


def ref_stats_cached(ref_feats, not_class):
    """(mu, sigma) of the allowed-class reference embeddings, memoised per not_class.

    Recomputing this is a 1280x1280 covariance over ~15k vectors -- identical for every folder
    sharing a not_class, so without the cache a 90-folder-per-class sweep does it 90 times.
    """
    if not_class not in _REF_STATS_CACHE:
        _REF_STATS_CACHE[not_class] = get_embeddings_statistics(ref_feats)
    return _REF_STATS_CACHE[not_class]


def score_against_class(gen_feats, seqs, preds, not_class, detector, reference_df, cache_dir,
                         skip_precision_recall=False):
    """The metrics that depend on which class is forbidden. Everything else about a pool -- its
    embeddings and its IgLM scores -- is computed once by the caller."""
    ref_feats = get_allowed_ref_feats(detector, reference_df, not_class, cache_dir)
    mu_r, sig_r = ref_stats_cached(ref_feats, not_class)
    mu_g, sig_g = get_embeddings_statistics(gen_feats)

    # k-NN precision/recall rebuilds the reference manifold from scratch on every call -- the
    # full |ref| x |ref| pairwise distance matrix (~15k x 15k here) just to get each reference
    # point's k-NN radius. That depends only on not_class, so across a sweep it is the same
    # handful of computations repeated once per folder, and it dominates the per-folder cost by
    # an order of magnitude. Neither metric is reported in the paper.
    precision = recall = float("nan")
    if not skip_precision_recall:
        pr = knn_precision_recall_features(ref_feats, gen_feats, nhood_sizes=[3])
        precision, recall = pr["precision"].item(), pr["recall"].item()

    return {
        "n_generated":    len(seqs),
        "n_parsed":       int(preds.notna().sum()),
        "violation_rate": float((preds == not_class).mean()),
        "fpd":            calculate_frechet_distance(mu_r, sig_r, mu_g, sig_g),
        "precision":      precision,
        "recall":         recall,
    }


def assemble_row(config, not_class, scores, iglm_mean, iglm_std, run_name, checkpoint):
    """One results row: the settings that produced the pool, then what it scored.

    The column order differs per CSV because the two describe different methods; fixing it here
    means re-running a sweep appends to an existing file rather than reshaping it.
    """
    tail = {"iglm_gen_ll_mean": iglm_mean, "iglm_gen_ll_std": iglm_std,
            "run_name": run_name, "checkpoint": checkpoint}
    if config["method"] == "dynamic":
        settings = {k: config[k] for k in (
            "not_class", "temp", "initial_guid_scale", "offset", "guid_scale_max",
            "num_steps", "scheduler_type", "use_prior_weighted_init")}
        return {**settings, **scores, **tail}
    return {"not_class": not_class, **scores,
            "method": config["method"], "guid_scale": config["guid_scale"],
            "num_steps": config["num_steps"], "scheduler_type": config["scheduler_type"],
            **tail}


def find_reused_unconditional_folder(run_name, checkpoint_number, num_steps, scheduler_type):
    """The guid_scale=0.0 pool from the natural-frequency sweep (run/sample_sweep.sh), reusable
    as the unconditional baseline instead of sampling it again. Returns (path, config) or None.

    That folder's samples.csv holds only the (ignored, since guid_scale=0) drawn label, not a
    classification, so the caller classifies it before scoring.
    """
    pattern = (f"logs/Figures/{run_name}/conditional/ckpt_{checkpoint_number}/"
               f"class_natural_gs_0.00_steps_{num_steps}_sched_{scheduler_type}_n_*")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    folder_path = matches[0]
    config = {"method": "unconditional", "not_class": None, "guid_scale": 0.0,
               "num_steps": num_steps, "scheduler_type": scheduler_type,
               "n_samples": int(folder_path.rsplit("_n_", 1)[1])}
    return folder_path, config


def main():
    parser = argparse.ArgumentParser(description="NOT-gate metrics: the dynamic gate and both baselines")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--to_evaluate_checkpoints", nargs="+", type=int, required=True)
    parser.add_argument("--baseline", action="store_true",
                         help="Score the baseline sweeps (unconditional / static negative guidance) "
                              "instead of the dynamic gate, into not_gate_baseline_results.csv.")
    parser.add_argument("--label_reference_csv", default="data/oas95_igblast_human/train.csv")
    parser.add_argument("--not_class", type=int, nargs="+", default=None,
                         help="Filter to folders with these forbidden class(es). Default: all.")
    parser.add_argument("--folder_name", default=None, help="Evaluate a single specific folder instead of discovering all.")
    parser.add_argument("--esm_model_name", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--max_ref_seqs", type=int, default=20000,
                         help="Load a uniform-random sample of N rows from --label_reference_csv instead of the full "
                              "multi-million-row file. Defaulted on -- pass 0 to load it in full.")
    parser.add_argument("--skip_iglm", action="store_true",
                         help="Skip the IgLM naturalness pass (iglm_gen_ll_mean/std -- the 'quality' half of the "
                              "accuracy/quality tradeoff). Faster if you only need violation_rate/fpd.")
    parser.add_argument("--iglm_model", default="IgLM", choices=["IgLM", "IgLM-S"])
    parser.add_argument("--chain_token", default="[HEAVY]", help="IgLM chain token")
    parser.add_argument("--species_token", default="[HUMAN]", help="IgLM species token")
    parser.add_argument("--force", action="store_true", help="Recompute metrics even for settings already in the results CSV.")
    parser.add_argument("--tag", default="",
                         help="Run tag matching sampling.not_tag: reads samples from "
                              "logs/Figures/{run_name}/not_gate[_baseline]_{tag}/ and writes to "
                              "logs/Results/{run_name}/{tag}/. Empty = the untagged layout.")
    parser.add_argument("--shard", type=int, default=None,
                         help="Shard index for parallel scoring. Each shard takes every Nth folder "
                              "(--num_shards) and writes its own CSV under {results_dir}/shards/, "
                              "merged afterwards by scripts/merge_metric_shards.py. REQUIRED for "
                              "concurrent jobs: the writer read-modify-writes the whole CSV per row, "
                              "so two jobs sharing one path silently drop each other's rows.")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards (see --shard).")
    parser.add_argument("--skip_precision_recall", action="store_true",
                         help="Skip k-NN precision/recall. It rebuilds the full |ref|x|ref| distance "
                              "matrix per folder and dominates runtime; neither metric is in the paper.")
    parser.add_argument("--skip_reuse_unconditional", action="store_true",
                         help="--baseline only: don't fall back to the natural-frequency guid_scale=0.0 "
                              "sweep for the unconditional row. Essential when using --tag: that sweep's "
                              "dedup key is identical to the one a tagged unconditional run produces, so "
                              "whichever is scored last would overwrite the other.")
    parser.add_argument("--unconditional_num_steps", type=int, default=64,
                         help="num_steps of the reused natural-frequency folder (see --skip_reuse_unconditional).")
    parser.add_argument("--unconditional_scheduler_type", default="linear")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = args.run_name
    samples_base = "not_gate_baseline" if args.baseline else "not_gate"
    results_name = "not_gate_baseline_results.csv" if args.baseline else "not_gate_results.csv"
    dedup_keys = DEDUP_KEYS[results_name]

    print(f"\nRun: {run_name}")
    print(f"Checkpoints: {args.to_evaluate_checkpoints}")

    # Passed explicitly rather than left to build_classifier's default, because the same value
    # has to reach reference_pool_tag below -- the subsample it selects is what the cached
    # reference embeddings are embeddings OF. This script exposes no --seed; 0 is the default
    # it always used.
    ref_seed = 0
    classify_fn, num_classes, class_names, reference_df = build_classifier(
        args.label_reference_csv, max_ref_seqs=args.max_ref_seqs or None, seed=ref_seed,
    )

    print("Loading ESM2 detector...")
    detector = ESM2Detector(model_name=args.esm_model_name, device=device)

    iglm_evaluator = None
    if not args.skip_iglm:
        print(f"Loading {args.iglm_model} evaluator...")
        iglm_evaluator = IgLMEvaluator(model_name=args.iglm_model)

    csv_path = tagged_results_path(run_name, results_name, tag=args.tag, shard=args.shard)
    print(f"Writing results to {csv_path}")

    existing_keys = set()
    if not args.force and os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        existing_keys = set(existing_df[dedup_keys].itertuples(index=False, name=None))
        print(f"Found {len(existing_keys)} settings already in {csv_path} -- these will be skipped (use --force to recompute).")

    all_results = []
    for checkpoint_number in args.to_evaluate_checkpoints:
        print(f"\n{'='*50}\nCheckpoint {checkpoint_number}\n{'='*50}")
        # Keyed by the model and the reference pool that produced the tensors: these are
        # embeddings of the allowed-class reference sequences, so --esm_model_name or
        # --max_ref_seqs changing while the path does not means the next run reads the previous
        # run's embeddings and reports FPD against a pool it never loaded. See
        # reference_pool_tag.
        cache_dir = os.path.join(
            "data/esm_embeddings_cache", model_tag(args.esm_model_name),
            reference_pool_tag(args.label_reference_csv, args.max_ref_seqs or None, ref_seed))

        # (folder_path, display name, config) for everything to score at this checkpoint.
        samples_dir = os.path.join(
            tagged_samples_root(run_name, samples_base, args.tag), f"ckpt_{checkpoint_number}")
        targets = []
        if os.path.isdir(samples_dir):
            names = [args.folder_name] if args.folder_name else sorted(os.listdir(samples_dir))
            for f in names:
                config = parse_folder(f)
                if config is not None and os.path.isdir(os.path.join(samples_dir, f)):
                    targets.append((os.path.join(samples_dir, f), f, config))
        else:
            print(f"  No samples dir at {samples_dir}.")

        if args.baseline and not args.skip_reuse_unconditional:
            if any(c["method"] == "unconditional" for _, _, c in targets):
                print("  Already have a directly-sampled 'unconditional' folder -- not reusing "
                      "the natural-frequency sweep too.")
            else:
                found = find_reused_unconditional_folder(
                    run_name, checkpoint_number, args.unconditional_num_steps,
                    args.unconditional_scheduler_type)
                if found is None:
                    print(f"  No reusable unconditional (natural-frequency, guid_scale=0.0, "
                          f"num_steps={args.unconditional_num_steps}) folder found -- run "
                          f"run/sample_sweep.sh first, or pass --unconditional_num_steps to "
                          f"match an existing sweep.")
                else:
                    path, config = found
                    targets.append((path, os.path.basename(path) + " [reused]", config))

        if args.not_class is not None:
            targets = [t for t in targets
                       if t[2]["not_class"] is None or t[2]["not_class"] in args.not_class]
        if args.shard is not None:
            # Stride, not a contiguous block: folders are name-sorted, so a block split would
            # hand one shard every folder of a single class, and the per-not_class reference
            # caches would still be rebuilt in every other shard.
            targets = targets[args.shard::args.num_shards]

        print(f"Found {len(targets)} folders to evaluate"
              + (f" (shard {args.shard}/{args.num_shards})" if args.shard is not None else ""))

        for folder_path, folder, config in targets:
            samples_csv = os.path.join(folder_path, "samples.csv")
            if not os.path.exists(samples_csv):
                print(f"    no samples.csv in {folder}, skipping")
                continue
            df = pd.read_csv(samples_csv)
            seqs = df["sequence"].astype(str).tolist()
            if "predicted_class" in df.columns:
                preds = df["predicted_class"]
            else:
                print(f"    {folder}: no predicted_class column -- classifying {len(seqs)} sequences now...")
                preds = pd.Series([classify_fn(s) for s in seqs])

            # Class-independent, so computed once for the pool even when it is scored against
            # every class below.
            gen_feats = get_embeddings_in_batches(detector, seqs, batch_size=64).cpu()
            iglm_mean, iglm_std = float("nan"), float("nan")
            if iglm_evaluator is not None:
                gen_lls = score_iglm(iglm_evaluator, seqs, args.chain_token, args.species_token,
                                      desc="IgLM-gen")
                iglm_mean, iglm_std = nan_mean_std(gen_lls)

            # The unconditional pool has no forbidden class of its own, so it is scored against
            # every class. The other methods fix one per folder.
            not_classes = range(num_classes) if config["not_class"] is None else [config["not_class"]]

            for not_class in not_classes:
                key = tuple(assemble_row(config, not_class, {}, iglm_mean, iglm_std,
                                          run_name, checkpoint_number)[k] for k in dedup_keys)
                if key in existing_keys:
                    print(f"  Skipping {folder} (not_class={not_class}) -- already in {csv_path}")
                    continue

                print(f"\n── {folder}" + (f"  (not_class={not_class})" if config["not_class"] is None else ""))
                scores = score_against_class(
                    gen_feats, seqs, preds, not_class, detector, reference_df, cache_dir,
                    skip_precision_recall=args.skip_precision_recall)
                row = assemble_row(config, not_class, scores, iglm_mean, iglm_std,
                                   run_name, checkpoint_number)
                # Provenance of what fpd was measured against -- see the same note in
                # metrics_sequences. Not dedup keys.
                row["esm_model"] = args.esm_model_name
                row["max_ref_seqs"] = args.max_ref_seqs or None
                row["ref_seed"] = ref_seed
                all_results.append(row)
                existing_keys.add(key)
                append_result_row(csv_path, row, dedup_keys)

                print(f"    viol={row['violation_rate']:.3f}  fpd={row['fpd']:.2f}  "
                      f"P={row['precision']:.3f}  R={row['recall']:.3f}  "
                      f"iglm_ll={row['iglm_gen_ll_mean']:.3f}  [saved to {csv_path}]")

    if not all_results:
        print("\nNo results computed.")
        return

    summary = ["checkpoint", "not_class", "violation_rate", "fpd", "iglm_gen_ll_mean"]
    summary[1:1] = ["method", "guid_scale"] if args.baseline else ["temp", "initial_guid_scale", "offset"]
    print(f"\nAll results saved incrementally to {csv_path}")
    print(pd.DataFrame(all_results)[summary].to_string(index=False))


if __name__ == "__main__":
    main()
