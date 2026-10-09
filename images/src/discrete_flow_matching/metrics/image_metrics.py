import torch

from .c2st import compute_c2st
from .frechet_inception_distance import calculate_frechet_distance, get_embeddings_statistics
from .precision_and_recall import knn_precision_recall_features


def image_metrics(ref_feats: torch.Tensor, gen_feats: torch.Tensor):
    """FID, k-NN precision/recall (k=3) and C2ST of generated vs. reference Inception features."""
    mu_ref, sigma_ref = get_embeddings_statistics(ref_feats)
    mu_gen, sigma_gen = get_embeddings_statistics(gen_feats)
    fid = calculate_frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen)

    pr = knn_precision_recall_features(ref_feats, gen_feats, nhood_sizes=[3])

    # C2ST on a random reference subset of the same size as the generated set
    idx = torch.randperm(ref_feats.shape[0])[:gen_feats.shape[0]]
    c2st_accuracy, c2st_log_loss = compute_c2st(ref_feats[idx], gen_feats)

    return {
        "fid": fid,
        "precision": pr["precision"].item(),
        "recall": pr["recall"].item(),
        "c2st_accuracy": c2st_accuracy,
        "c2st_log_loss": c2st_log_loss,
    }
