"""
Add predicted_class/correct columns to the samples.csv files written by sample_sequences.py, by
classifying each generated sequence's V-gene family with nearest-neighbour framework identity
against the full reference pool (VGeneFamilyClassifier in sampling_utils.py).

Kept as a separate opt-in step rather than run inside the sampler: it is a brute-force
O(n_generated * n_reference) scan, which takes hours against a multi-million-row pool. Run it
only when you need per-class accuracy.

Reads / modifies in place:
        logs/Figures/<run>/conditional/ckpt_<k>/<one folder per setting>/samples.csv

Example:
  python -m scripts.classify_vgene_samples \\
      --run_name oas95_igblast_human_35M_uniform \\
      --to_evaluate_checkpoints 20 \\
      --label_reference_csv data/oas95_igblast_human/train.csv
"""
import argparse
import os

import pandas as pd
from tqdm import tqdm

from scripts.sampling_utils import build_classifier, parse_folder


def classify_folder(folder_path, classify_fn, target_class, force):
    csv_path = os.path.join(folder_path, "samples.csv")
    if not os.path.exists(csv_path):
        print(f"    no samples.csv in {folder_path}, skipping")
        return None

    df = pd.read_csv(csv_path)
    if "predicted_class" in df.columns and not force:
        print(f"    already classified, skipping (use --force to recompute): {csv_path}")
        return df

    seqs = df["sequence"].astype(str).tolist()
    preds = [classify_fn(seq) for seq in tqdm(seqs, desc="  classifying", leave=False)]
    df["predicted_class"] = preds
    df["correct"] = [p == target_class for p in preds]
    df.to_csv(csv_path, index=False)

    n_parsed = df["predicted_class"].notna().sum()
    n_correct = df["correct"].sum()
    print(f"    {n_correct}/{len(df)} classified as target class ({n_parsed} parsed) -> {csv_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Classify generated V-gene samples against the full reference pool.")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--to_evaluate_checkpoints", nargs="+", type=int, required=True)
    parser.add_argument("--label_reference_csv", default="data/oas95_igblast_human/train.csv")
    parser.add_argument("--max_ref_seqs", type=int, default=20000,
                         help="Load a uniform-random sample of N rows from --label_reference_csv instead of the full "
                              "multi-million-row file -- classify_fn scans this pool once per generated sequence, so "
                              "its size directly drives this script's runtime. Defaulted on -- pass 0 to load in full.")
    parser.add_argument("--target_class", type=int, default=None,
                         help="Only classify folders with this target class. If omitted, all classes are done.")
    parser.add_argument("--folder_name", default=None, help="Classify a single specific folder instead of discovering all.")
    parser.add_argument("--force", action="store_true", help="Recompute even if predicted_class is already present.")
    args = parser.parse_args()

    print(f"\nRun: {args.run_name}")
    print(f"Checkpoints: {args.to_evaluate_checkpoints}")
    print("Building V-gene classifier (one-time reference-pool scan)...")
    classify_fn, _, _, _ = build_classifier(args.label_reference_csv, max_ref_seqs=args.max_ref_seqs or None)

    for checkpoint_number in args.to_evaluate_checkpoints:
        samples_dir = f"logs/Figures/{args.run_name}/conditional/ckpt_{checkpoint_number}"
        if not os.path.isdir(samples_dir):
            print(f"\nCheckpoint {checkpoint_number}: no samples dir at {samples_dir}, skipping.")
            continue

        print(f"\n{'='*50}\nCheckpoint {checkpoint_number}\n{'='*50}")

        if args.folder_name:
            folders = [args.folder_name]
        else:
            folders = sorted([
                f for f in os.listdir(samples_dir)
                if os.path.isdir(os.path.join(samples_dir, f)) and parse_folder(f) is not None
            ])
        if args.target_class is not None:
            folders = [f for f in folders if (c := parse_folder(f)) is not None and c["class"] == args.target_class]

        print(f"Found {len(folders)} folder(s) to classify")
        for folder in folders:
            config = parse_folder(folder)
            if config is None:
                continue
            print(f"\n-- {folder}")
            classify_folder(os.path.join(samples_dir, folder), classify_fn, config["class"], args.force)


if __name__ == "__main__":
    main()
