import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance


def list_image_paths(imgs_path, num_samples=None):
    """Sorted paths of the .png files in imgs_path, optionally truncated to num_samples."""
    image_paths = sorted([os.path.join(imgs_path, f) for f in os.listdir(imgs_path) if f.endswith(".png")])

    if num_samples is not None:
        image_paths = image_paths[:num_samples]
    return image_paths


def load_image(path):
    """Load an image as a uint8 [3, H, W] tensor."""
    img = np.array(Image.open(path).convert("RGB"))
    return torch.from_numpy(img).permute(2, 0, 1).to(torch.uint8)


@torch.no_grad()
def get_image_embeddings_in_batches(detector, image_paths, batch_size=64, device="cuda"):
    feats = []
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i + batch_size]
        batch_imgs = torch.stack([load_image(p) for p in batch_paths]).to(device)
        feats.append(detector(batch_imgs).cpu())
    return torch.cat(feats, dim=0)


class InceptionV3Detector:
    """2048-d pool features of the standard FID InceptionV3 network."""

    def __init__(self, device="cuda"):
        self.device = device
        self.model = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
        self.feature_dim = 2048

    @torch.no_grad()
    def __call__(self, x):
        """x: [B, 3, H, W] in [0, 255]; resizing and normalisation are handled by torchmetrics."""
        features = self.model.inception(x)  # [B, 2048, 1, 1]
        return features.squeeze(-1).squeeze(-1)
