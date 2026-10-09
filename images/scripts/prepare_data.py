"""Download MNIST to data/<name>/ and export the training images as PNGs.

The PNGs in data/<name>/train/images/ are the reference set for FID / precision / recall.

Usage (from the Image_experiments/ root):
    python -m scripts.prepare_data

Other Hugging Face image datasets can be added to HF_NAMES / IMAGE_COLUMNS.
"""
import os
import sys

from datasets import load_dataset, load_from_disk
from tqdm import tqdm

HF_NAMES = {"mnist": "ylecun/mnist"}
IMAGE_COLUMNS = {"mnist": "image"}


def prepare(name):
    out_dir = f"data/{name}"
    if not os.path.exists(os.path.join(out_dir, "dataset_dict.json")):
        load_dataset(HF_NAMES[name]).save_to_disk(out_dir)
    ds = load_from_disk(out_dir)["train"]

    img_dir = os.path.join(out_dir, "train", "images")
    os.makedirs(img_dir, exist_ok=True)
    for idx, sample in enumerate(tqdm(ds, desc=f"{name}: exporting PNGs")):
        sample[IMAGE_COLUMNS[name]].save(os.path.join(img_dir, f"{idx:06d}.png"))
    print(f"{name}: {len(ds)} images in {img_dir}")


if __name__ == "__main__":
    for name in sys.argv[1:] or list(HF_NAMES):
        prepare(name)
