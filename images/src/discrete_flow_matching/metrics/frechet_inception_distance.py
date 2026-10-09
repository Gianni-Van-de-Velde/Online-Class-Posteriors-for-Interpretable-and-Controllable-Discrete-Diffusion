"""Fréchet distance between Gaussian fits of two feature sets (as in pytorch-fid / TTUR)."""

import numpy as np
from scipy import linalg


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Fréchet distance between N(mu1, sigma1) and N(mu2, sigma2).
    mu1, sigma1: mean and covariance of the reference features
    mu2, sigma2: mean and covariance of the generated features
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # If sqrtm is numerically unstable, regularise the covariances and recompute
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component in covmean: {m:.4f}')
        covmean = covmean.real

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


def get_embeddings_statistics(embeddings):
    """Mean and covariance of a [num_samples, embedding_dim] tensor."""
    embeds_np = embeddings.detach().cpu().numpy()
    mu = np.mean(embeds_np, axis=0)
    sigma = np.cov(embeds_np, rowvar=False)
    return mu, sigma
