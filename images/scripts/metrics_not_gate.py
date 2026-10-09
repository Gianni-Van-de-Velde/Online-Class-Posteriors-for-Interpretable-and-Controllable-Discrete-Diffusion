"""
Compute metrics for the NOT-gate samples written by sample_not_gate.py (NOT gate and baseline).

Image quality (FID, precision, recall, C2ST) is measured against the real images of the allowed
classes, i.e. all classes except the avoided one. A pretrained ViT classifier gives:
  - violation_rate   fraction of samples classified as the avoided class
  - kl_from_uniform  KL between the predicted class distribution and the target (uniform over
                     the allowed classes, zero on the avoided class)
  - mean_confidence  mean classifier confidence
  - fid_class_<c>    FID of the samples classified as c against the real images of class c

Reads:  logs/Figures/<run>/not_gate/ckpt_<k>/<one folder per setting>/
Writes: logs/Results/<run>/not_gate_results.csv, one row per setting
        (settings already in the CSV are skipped unless --force)
"""
import os
import re
import argparse
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk
from scipy.stats import entropy

from discrete_flow_matching.metrics import (
    list_image_paths,
    get_image_embeddings_in_batches,
    InceptionV3Detector,
    image_metrics,
    calculate_frechet_distance,
    get_embeddings_statistics,
)
from scripts.image_utils import get_classifier

# e.g. not_0_temp_0.0100_initgs_0.5000_gsmax_5.00_sampler_ode_sched_cosine_n_4096
NOT_FOLDER_RE = re.compile(
    r"^not_(\d+)_temp_([0-9.]+)_initgs_([0-9.]+)_gsmax_([0-9.]+)_sampler_(ancestral|ode)"
    r"_sched_(\w+)_n_(\d+)(_denoise_ancestral)?$"
)
# e.g. not_baseline_0_lam_0.2000_sched_cosine_n_4096
BASELINE_FOLDER_RE = re.compile(r"^not_baseline_(\d+)_lam_([0-9.]+)_sched_(\w+)_n_(\d+)(_denoise_ancestral)?$")

KEY_COLUMNS = ["run_name", "checkpoint", "not_class", "is_baseline", "temp", "initial_guid_scale",
               "guid_scale_max", "lam_constant", "sampler_type", "scheduler_type", "denoise_ancestral"]


def parse_folder(name):
    """Sampling settings from the folder name, or None if it is not a NOT-gate sample folder."""
    m = NOT_FOLDER_RE.match(name)
    if m:
        return {
            "not_class": int(m.group(1)),
            "is_baseline": False,
            "temp": float(m.group(2)),
            "initial_guid_scale": float(m.group(3)),
            "guid_scale_max": float(m.group(4)),
            "lam_constant": None,
            "sampler_type": m.group(5),             # kernel used to score the posterior update
            "scheduler_type": m.group(6),
            "denoise_ancestral": m.group(8) is not None,
            "num_samples": int(m.group(7)),
        }
    m = BASELINE_FOLDER_RE.match(name)
    if m:
        return {
            "not_class": int(m.group(1)),
            "is_baseline": True,
            "temp": None,
            "initial_guid_scale": None,
            "guid_scale_max": None,
            "lam_constant": float(m.group(2)),
            "sampler_type": "baseline",
            "scheduler_type": m.group(3),
            "denoise_ancestral": m.group(5) is not None,
            "num_samples": int(m.group(4)),
        }
    return None


def get_ref_feats(detector, dataset_name, classes, device, batch_size=128):
    """Inception features of the training images of the given classes (cached)."""
    cache_path = f"data/{dataset_name}/train/inception_embeddings_classes_{'_'.join(map(str, sorted(classes)))}.pt"
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    ds = load_from_disk(f"data/{dataset_name}")["train"]
    img_key = "img" if "img" in ds[0] else "image"
    imgs = []
    for ex in ds:
        if ex["label"] in classes:
            t = torch.from_numpy(np.array(ex[img_key]))
            imgs.append(t.unsqueeze(0).expand(3, -1, -1) if t.ndim == 2 else t.permute(2, 0, 1))  # -> [3, H, W]
    imgs = torch.stack(imgs)

    feats = torch.cat([detector(imgs[i:i + batch_size].to(device)).cpu()
                       for i in tqdm(range(0, len(imgs), batch_size), desc="reference features")])
    torch.save(feats, cache_path)
    return feats


def compute_metrics(folder_path, settings, inception, classifier, dataset_name, num_classes, device):
    """One CSV row for one sample folder."""
    img_paths = list_image_paths(folder_path, num_samples=settings["num_samples"])
    if not img_paths:
        print(f"    no images found in {folder_path}, skipping")
        return None
    not_class = settings["not_class"]
    allowed = [c for c in range(num_classes) if c != not_class]

    gen_feats = get_image_embeddings_in_batches(inception, img_paths, batch_size=128, device=device).cpu()
    ref_feats = get_ref_feats(inception, dataset_name, allowed, device)
    row = {k: v for k, v in settings.items() if k != "num_samples"}
    row.update(n_generated=len(img_paths), **image_metrics(ref_feats, gen_feats))

    # Classifier-based metrics
    pred, conf, _ = classifier.predict(img_paths)
    pred, conf = pred.numpy(), conf.numpy()
    row["violation_rate"] = float((pred == not_class).mean())
    row["mean_confidence"] = float(conf.mean())

    pred_dist = np.bincount(pred, minlength=num_classes) / len(pred)
    target_dist = np.full(num_classes, 1.0 / len(allowed))
    target_dist[not_class] = 1e-8          # avoid an infinite KL when violations occur
    target_dist /= target_dist.sum()
    row["kl_from_uniform"] = float(entropy(pred_dist + 1e-8, target_dist))

    # Per-class FID (needs at least 10 samples of that class)
    for c in allowed:
        mask = pred == c
        if mask.sum() < 10:
            row[f"fid_class_{c}"] = float("nan")
            continue
        mu_r, sig_r = get_embeddings_statistics(get_ref_feats(inception, dataset_name, [c], device))
        mu_g, sig_g = get_embeddings_statistics(gen_feats[mask])
        row[f"fid_class_{c}"] = calculate_frechet_distance(mu_r, sig_r, mu_g, sig_g)
    return row


def main():
    parser = argparse.ArgumentParser(description="Metrics for NOT-gate and baseline samples")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--dataset_name", default="mnist")
    parser.add_argument("--to_evaluate_checkpoints", nargs="+", type=int, required=True)
    parser.add_argument("--not_class", type=int, default=None, help="Evaluate only this avoided class.")
    parser.add_argument("--folder_name", default=None, help="Evaluate only this sample folder.")
    parser.add_argument("--force", action="store_true", help="Recompute settings already in the CSV.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    csv_path = f"logs/Results/{args.run_name}/not_gate_results.csv"
    done = set()
    if os.path.exists(csv_path) and not args.force:
        done = set(pd.read_csv(csv_path)[KEY_COLUMNS].itertuples(index=False, name=None))

    inception = InceptionV3Detector(device=device)
    classifier = get_classifier(args.dataset_name, device=device)

    for ckpt in args.to_evaluate_checkpoints:
        samples_dir = f"logs/Figures/{args.run_name}/not_gate/ckpt_{ckpt}"
        if not os.path.isdir(samples_dir):
            print(f"No samples for checkpoint {ckpt} in {samples_dir}, skipping.")
            continue
        folders = [args.folder_name] if args.folder_name else sorted(os.listdir(samples_dir))

        for folder in folders:
            settings = parse_folder(folder)
            if settings is None or (args.not_class is not None and settings["not_class"] != args.not_class):
                continue
            key = (args.run_name, ckpt, *(settings[k] for k in KEY_COLUMNS[2:]))
            if key in done:
                print(f"  {folder}: already in {csv_path}, skipping")
                continue

            print(f"  {folder}")
            row = compute_metrics(os.path.join(samples_dir, folder), settings, inception, classifier,
                                  args.dataset_name, classifier.num_classes, device)
            if row is None:
                continue
            row.update(run_name=args.run_name, checkpoint=ckpt)
            done.add(key)
            print(f"    FID={row['fid']:.2f}  P={row['precision']:.3f}  R={row['recall']:.3f}"
                  f"  C2ST_acc={row['c2st_accuracy']:.3f}  violation={row['violation_rate']:.3f}")

            # Save after every folder: a full sweep takes hours and should survive interruptions
            df = pd.DataFrame([row])
            if os.path.exists(csv_path):
                df = pd.concat([pd.read_csv(csv_path), df], ignore_index=True)
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            df.drop_duplicates(subset=KEY_COLUMNS, keep="last").to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
