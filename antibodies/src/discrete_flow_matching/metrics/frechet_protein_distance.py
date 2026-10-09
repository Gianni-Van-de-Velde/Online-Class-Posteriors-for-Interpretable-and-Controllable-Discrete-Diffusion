"""Fréchet distance between Gaussian fits of two embedding sets (FPD, the protein analogue of FID)."""

import numpy as np
import torch
from scipy import linalg


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    """
    Fréchet distance between N(mu1, sigma1) and N(mu2, sigma2).
    mu1, sigma1: mean and covariance of the reference embeddings
    mu2, sigma2: mean and covariance of the generated embeddings

    NB: where sqrtm comes back complex this simply takes the real part. The image
    experiments' counterpart instead regularises the covariances and recomputes; the two
    therefore differ slightly on ill-conditioned inputs. Kept as is for reproducibility.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # sqrtm can return complex values due to floating-point error; take the real part
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


def get_embeddings_statistics(embeddings: torch.Tensor):
    """Mean and covariance of a [num_samples, embedding_dim] tensor."""
    embeds_np = embeddings.detach().cpu().numpy()
    mu = np.mean(embeds_np, axis=0)
    sigma = np.cov(embeds_np, rowvar=False)
    return mu, sigma
