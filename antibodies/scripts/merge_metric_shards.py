"""
Merges the per-shard metrics CSVs written by metrics_not_gate.py --shard (with and without
--baseline) back into one CSV per results directory.

Sharding exists because those scripts read-modify-write the whole results CSV once per scored
folder, so N concurrent jobs pointed at one path silently drop each other's rows. Each shard
writes logs/Results/{run_name}/{tag}/shards/{name}.shard{k}.csv instead; this merges them back
to logs/Results/{run_name}/{tag}/{name}.csv.

Example:
  python -m scripts.merge_metric_shards \\
      --run_name oas95_igblast_human_35M_uniform --tag final
"""
import argparse
import glob
import os

import pandas as pd

# Must match the dedup keys in the two metrics scripts, otherwise merging could keep two rows
# describing the same setting.
DEDUP_KEYS = {
    "not_gate_results.csv": ["run_name", "checkpoint", "not_class", "temp", "initial_guid_scale",
                              "offset", "guid_scale_max", "num_steps", "scheduler_type",
                              "use_prior_weighted_init"],
    "not_gate_baseline_results.csv": ["run_name", "checkpoint", "method", "not_class",
                                       "guid_scale", "num_steps", "scheduler_type"],
}


def merge_one(results_dir, name, overwrite):
    shard_glob = os.path.join(results_dir, "shards", f"{os.path.splitext(name)[0]}.shard*.csv")
    shards = sorted(glob.glob(shard_glob))
    if not shards:
        print(f"  {name}: no shards at {shard_glob} -- skipping.")
        return
    out_path = os.path.join(results_dir, name)
    frames = [pd.read_csv(p) for p in shards]
    print(f"  {name}: {len(shards)} shards, {sum(len(f) for f in frames)} rows")

    # An existing merged file is folded in first and the shards win on conflict, so re-running
    # after a partial shard finishes tops the file up rather than discarding earlier work.
    if os.path.exists(out_path) and not overwrite:
        frames.insert(0, pd.read_csv(out_path))

    combined = pd.concat(frames, ignore_index=True)
    keys = [k for k in DEDUP_KEYS[name] if k in combined.columns]
    before = len(combined)
    combined = combined.drop_duplicates(subset=keys, keep="last")
    combined.to_csv(out_path, index=False)
    print(f"    -> {out_path}  ({len(combined)} rows, {before - len(combined)} duplicates dropped)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--tag", default="", help="Run tag used by the sampling/metrics jobs.")
    parser.add_argument("--overwrite", action="store_true",
                         help="Ignore any existing merged CSV instead of folding it in first.")
    args = parser.parse_args()

    results_dir = f"logs/Results/{args.run_name}" + (f"/{args.tag}" if args.tag else "")
    if not os.path.isdir(results_dir):
        raise SystemExit(f"No results dir at {results_dir}")

    print(f"Merging shards under {results_dir}")
    for name in DEDUP_KEYS:
        merge_one(results_dir, name, args.overwrite)


if __name__ == "__main__":
    main()
