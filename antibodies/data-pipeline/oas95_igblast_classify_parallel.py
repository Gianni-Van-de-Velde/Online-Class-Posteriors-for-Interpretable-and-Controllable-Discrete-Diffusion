#!/usr/bin/env python
"""Orchestrate N parallel oas95_igblast_classify.py workers over disjoint row ranges of
an OAS95 csv, then merge their outputs into one dataset (matching the
name,sequence,family_str,label,total_length format used throughout this repo) plus a
90/5 (or given) train/val split and a conf/dataset/<name>.yaml.

Each worker is single-threaded IgBLAST (embarrassingly parallel across independent
sequences scales far better than igblastp's own -num_threads at this small germline-db
size, per local benchmarking: ~1.7x from -num_threads=8 vs. expected near-linear scaling
across independent processes). Workers are individually resumable (same progress-file
mechanism as the sequential script), so killing and re-running this orchestrator picks up
where each worker left off instead of restarting from scratch.

Usage:
  python oas95_igblast_classify_parallel.py <csv_path> <out_dir> <total_rows> <num_workers>
                                            [--val-frac 0.02] [--val-subsample 20000]

Writes into <out_dir>:
  full.csv     every classified sequence (name,sequence,family_str,label,total_length)
  train.csv    (name,sequence,label) -- what conf/dataset/oas95_igblast_human_train.yaml reads
  val.csv      (name,sequence,label)
  val_20k.csv  label-stratified subsample of val.csv, for the dense-eval training config
  shards/      per-worker partial outputs + resume state + logs (safe to delete when done)
"""
import os
import sys
import glob
import argparse
import subprocess

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER_SCRIPT = os.path.join(HERE, "oas95_igblast_classify.py")
BUILD_INDEX_SCRIPT = os.path.join(HERE, "build_row_index.py")


def ensure_row_index(csv_path, out_prefix):
    """Build (once) or reuse the byte-offset index for csv_path -- lets every worker
    seek() to its start row instead of each independently scanning every row before it
    (the resource contention that killed most workers on 32- and 8-CPU attempts)."""
    index_path = f"{out_prefix}_rowindex.npy"
    if os.path.exists(index_path):
        print(f"Reusing existing row index at {index_path}", flush=True)
        return index_path
    print(f"Building row-offset index for {csv_path} -> {index_path} (one-time pass)...", flush=True)
    subprocess.run([sys.executable, BUILD_INDEX_SCRIPT, csv_path, index_path], check=True)
    return index_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="OAS95 train.csv (or test.csv)")
    ap.add_argument("out_dir", help="dataset directory to write, e.g. data/oas95_igblast_human")
    ap.add_argument("total_rows", type=int, help="rows in csv_path (wc -l)")
    ap.add_argument("num_workers", type=int)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--val-subsample", type=int, default=20000,
                    help="rows in the stratified val_20k.csv (0 to skip it)")
    args = ap.parse_args()

    csv_path, total_rows, num_workers = args.csv_path, args.total_rows, args.num_workers
    val_frac = args.val_frac
    out_dir = args.out_dir
    shard_dir = os.path.join(out_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    out_prefix = os.path.join(shard_dir, "shard")

    row_index_path = ensure_row_index(csv_path, out_prefix)

    rows_per_worker = -(-total_rows // num_workers)  # ceil div
    ranges = []
    for w in range(num_workers):
        start = w * rows_per_worker
        end = min(start + rows_per_worker, total_rows)
        if start >= end:
            break
        ranges.append((start, end))

    print(f"Splitting {total_rows} rows across {len(ranges)} workers "
          f"(~{rows_per_worker} rows/worker).", flush=True)

    log_dir = os.path.join(shard_dir, "worker_logs")
    os.makedirs(log_dir, exist_ok=True)

    procs = []
    for w, (start, end) in enumerate(ranges):
        worker_prefix = f"{out_prefix}_w{w}"
        log_path = os.path.join(log_dir, f"worker_{w}.log")
        cmd = [
            sys.executable, WORKER_SCRIPT, csv_path, worker_prefix,
            "--start-row", str(start), "--end-row", str(end), "--num-threads", "1",
            "--row-index", row_index_path,
        ]
        log_f = open(log_path, "w")
        p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((w, p, log_f))
        print(f"  worker {w}: rows [{start}, {end}) -> {worker_prefix}_full.csv (log: {log_path})", flush=True)

    failed = []
    for w, p, log_f in procs:
        ret = p.wait()
        log_f.close()
        if ret != 0:
            failed.append(w)
            print(f"  worker {w} FAILED (exit {ret}) -- see {log_dir}/worker_{w}.log", flush=True)
        else:
            print(f"  worker {w} finished OK.", flush=True)

    if failed:
        print(f"WARNING: {len(failed)} worker(s) failed: {failed}. "
              f"Re-run this same command to resume/retry just those (existing progress is kept).", flush=True)

    # Merge whatever worker outputs exist (even if some workers are still incomplete --
    # re-running the orchestrator later will pick up the rest via each worker's own resume).
    worker_csvs = sorted(glob.glob(f"{out_prefix}_w*_full.csv"))
    if not worker_csvs:
        print("No worker output files found -- nothing to merge.", flush=True)
        return

    print(f"Merging {len(worker_csvs)} worker output files...", flush=True)
    dfs = [pd.read_csv(p) for p in worker_csvs]
    full = pd.concat(dfs, ignore_index=True)
    before = len(full)
    full = full.drop_duplicates(subset="sequence").reset_index(drop=True)
    print(f"Merged: {before} rows -> {len(full)} after dropping {before - len(full)} exact-duplicate sequences.")

    full_path = os.path.join(out_dir, "full.csv")
    full.to_csv(full_path, index=False)

    shuffled = full.sample(frac=1.0, random_state=0).reset_index(drop=True)
    n_val = max(1, int(val_frac * len(shuffled)))
    val, train = shuffled.iloc[:n_val], shuffled.iloc[n_val:]
    train_path = os.path.join(out_dir, "train.csv")
    val_path = os.path.join(out_dir, "val.csv")
    train[["name", "sequence", "label"]].to_csv(train_path, index=False)
    val[["name", "sequence", "label"]].to_csv(val_path, index=False)

    # Stratified validation subsample for conf/dataset/oas95_igblast_human_train_denseval.yaml:
    # the trainer's evaluate() walks the WHOLE validation dataloader with no early break, so a
    # 1.1M-row split costs ~14x the training compute at eval_steps=100. Stratifying on `label`
    # keeps the rare families' true share (IGHV7 0.0047 -> 0.0046, IGHV6 0.0061 -> 0.0060).
    sub_path = None
    if args.val_subsample and len(val) > args.val_subsample:
        frac = args.val_subsample / len(val)
        rng, idx = np.random.default_rng(0), []
        for _, g in val.groupby("label"):
            idx.extend(rng.choice(g.index.to_numpy(),
                                  size=max(1, round(len(g) * frac)), replace=False))
        sub_path = os.path.join(out_dir, "val_20k.csv")
        val.loc[sorted(idx), ["name", "sequence", "label"]].to_csv(sub_path, index=False)

    fams = [f"IGHV{i}" for i in range(1, 8)]
    print("\nPer-family distribution:")
    print(full["family_str"].value_counts().reindex(fams, fill_value=0))
    print(f"\nFinal: {full_path} ({len(full)} rows), "
          f"{train_path} ({len(train)}), {val_path} ({len(val)})"
          + (f", {sub_path}" if sub_path else ""))
    print(f"Shards kept in {shard_dir} for resume -- safe to delete once you are done.")


if __name__ == "__main__":
    main()
