"""
Compute FID, precision, recall and C2ST for the samples written by sample_images.py.

Reads:  logs/Figures/<run>/samples/ckpt_<k>/<one folder per sampling setting>/
Writes: logs/Results/<run>/sample_metrics.csv, one row per setting
        (settings already in the CSV are skipped unless --force)
"""
import os
import re
import argparse
import torch
import pandas as pd

from discrete_flow_matching.metrics import (
    list_image_paths,
    get_image_embeddings_in_batches,
    InceptionV3Detector,
    image_metrics,
)

# e.g. guid_1.0_num_steps_128_final_samples_4096_cosine (no scheduler suffix = linear)
FOLDER_RE = re.compile(r"guid_([0-9.]+)_num_steps_([0-9]+)_final_samples_([0-9]+)(?:_([a-z_]+))?")
KEY_COLUMNS = ["run_name", "checkpoint", "num_steps", "guid_scale", "scheduler_type"]


def parse_folder(name):
    """Sampling settings from the folder name, or None if it is not a sample folder."""
    m = FOLDER_RE.search(name)
    if m is None:
        return None
    guid_scale, num_steps, num_samples, scheduler = m.groups()
    return {
        "guid_scale": float(guid_scale),
        "num_steps": int(num_steps),
        "num_samples": int(num_samples),
        "scheduler_type": scheduler or "linear",
    }


def get_ref_feats(detector, dataset_name):
    """Inception features of the training images in data/<dataset>/train/images/ (cached)."""
    cache_path = f"data/{dataset_name}/train/inception_embeddings.pt"
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu")
    ref_imgs = list_image_paths(f"data/{dataset_name}/train/images")
    feats = get_image_embeddings_in_batches(detector, ref_imgs, batch_size=64).cpu()
    torch.save(feats, cache_path)
    return feats


def compute_metrics(folder_path, settings, inception, ref_feats):
    """One CSV row for one sample folder."""
    gen_imgs = list_image_paths(folder_path, num_samples=settings["num_samples"])
    if not gen_imgs:
        print(f"    no images found in {folder_path}, skipping")
        return None
    gen_feats = get_image_embeddings_in_batches(inception, gen_imgs, batch_size=64).cpu()
    return {**settings, **image_metrics(ref_feats, gen_feats)}


def main():
    parser = argparse.ArgumentParser(description="FID, precision, recall and C2ST of generated samples")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--dataset_name", default="mnist")
    parser.add_argument("--to_evaluate_checkpoints", nargs="+", type=int, required=True)
    parser.add_argument("--diffusion_type", default="uniform", help="Only stored in the CSV.")
    parser.add_argument("--folder_name", default=None, help="Evaluate only this sample folder.")
    parser.add_argument("--force", action="store_true", help="Recompute settings already in the CSV.")
    args = parser.parse_args()

    csv_path = f"logs/Results/{args.run_name}/sample_metrics.csv"
    done = set()
    if os.path.exists(csv_path) and not args.force:
        done = set(pd.read_csv(csv_path)[KEY_COLUMNS].itertuples(index=False, name=None))

    inception = InceptionV3Detector()
    ref_feats = get_ref_feats(inception, args.dataset_name)

    rows = []
    for ckpt in args.to_evaluate_checkpoints:
        samples_dir = f"logs/Figures/{args.run_name}/samples/ckpt_{ckpt}"
        if not os.path.isdir(samples_dir):
            print(f"No samples for checkpoint {ckpt} in {samples_dir}, skipping.")
            continue
        folders = [args.folder_name] if args.folder_name else sorted(os.listdir(samples_dir))

        for folder in folders:
            settings = parse_folder(folder)
            if settings is None:
                continue
            key = (args.run_name, ckpt, settings["num_steps"], settings["guid_scale"], settings["scheduler_type"])
            if key in done:
                print(f"  {folder}: already in {csv_path}, skipping")
                continue

            print(f"  {folder}")
            row = compute_metrics(os.path.join(samples_dir, folder), settings, inception, ref_feats)
            if row is None:
                continue
            row.update(run_name=args.run_name, checkpoint=ckpt, diffusion_type=args.diffusion_type)
            print(f"    FID={row['fid']:.2f}  P={row['precision']:.3f}  R={row['recall']:.3f}"
                  f"  C2ST_acc={row['c2st_accuracy']:.3f}")
            rows.append(row)

    if not rows:
        print("No results computed.")
        return

    # Append to the CSV; a recomputed setting replaces its old row
    df = pd.DataFrame(rows)
    if os.path.exists(csv_path):
        df = pd.concat([pd.read_csv(csv_path), df], ignore_index=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.drop_duplicates(subset=KEY_COLUMNS, keep="last").to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
