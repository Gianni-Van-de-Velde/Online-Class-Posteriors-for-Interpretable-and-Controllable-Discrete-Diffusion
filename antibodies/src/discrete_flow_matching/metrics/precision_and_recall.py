# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
# PyTorch port of https://github.com/kynkaat/improved-precision-and-recall-metric

"""k-NN precision and recall (Kynkäänniemi et al., 2019)."""

import numpy as np
import torch


def batch_pairwise_distances(U, V):
    """Squared Euclidean distances between the rows of U [N, D] and V [M, D]."""
    U_sq = (U ** 2).sum(dim=1, keepdim=True)    # [N, 1]
    V_sq = (V ** 2).sum(dim=1, keepdim=True).T  # [1, M]
    dist = U_sq + V_sq - 2 * U @ V.T
    return dist.clamp(min=0)


class DistanceBlock(torch.nn.Module):
    """Computes pairwise distances between two batches on the available GPUs."""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.DataParallel(self._DistanceModule())

    class _DistanceModule(torch.nn.Module):
        def forward(self, batch1, batch2):
            return batch_pairwise_distances(batch1, batch2)

    def pairwise_distances(self, U, V):
        """U: [N, D], V: [M, D] -> [N, M] numpy array of squared distances."""
        U = torch.tensor(U, dtype=torch.float32).cuda()
        V = torch.tensor(V, dtype=torch.float32).cuda()
        return self.model(U, V).cpu().numpy()


class ManifoldEstimator:
    """Estimates the manifold of a set of feature vectors as a union of k-NN hyperspheres."""

    def __init__(self, distance_block, features, row_batch_size=25000, col_batch_size=50000,
                 nhood_sizes=[3], clamp_to_percentile=None, eps=1e-5):
        """
        Args:
            distance_block: DistanceBlock used for the pairwise distance computations.
            features: [num_samples, D] feature vectors whose manifold is estimated.
            row_batch_size, col_batch_size: batch sizes for the pairwise distances
                (trade-off between memory usage and speed).
            nhood_sizes (list): neighbourhood sizes k used to estimate the manifold.
            clamp_to_percentile (float): prune hyperspheres whose radius exceeds this percentile.
            eps (float): small number for numerical stability.
        """
        num_samples = features.shape[0]
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.eps = eps
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self._ref_features = features
        self._distance_block = distance_block

        # Radius of each sample's hypersphere: distance to its k-th nearest neighbour.
        self.D = np.zeros([num_samples, self.num_nhoods], dtype=np.float32)
        distance_batch = np.zeros([row_batch_size, num_samples], dtype=np.float32)
        seq = np.arange(max(self.nhood_sizes) + 1, dtype=np.int32)

        for begin1 in range(0, num_samples, row_batch_size):
            end1 = min(begin1 + row_batch_size, num_samples)
            row_batch = features[begin1:end1]

            for begin2 in range(0, num_samples, col_batch_size):
                end2 = min(begin2 + col_batch_size, num_samples)
                col_batch = features[begin2:end2]
                distance_batch[0:end1-begin1, begin2:end2] = self._distance_block.pairwise_distances(row_batch, col_batch)

            self.D[begin1:end1, :] = np.partition(distance_batch[0:end1-begin1, :], seq, axis=1)[:, self.nhood_sizes]

        if clamp_to_percentile is not None:
            max_distances = np.percentile(self.D, clamp_to_percentile, axis=0)
            self.D[self.D > max_distances] = 0

    def evaluate(self, eval_features, return_realism=False, return_neighbors=False):
        """For each eval feature, whether it lies inside any reference hypersphere."""
        num_eval_samples = eval_features.shape[0]
        num_ref_samples = self.D.shape[0]
        distance_batch = np.zeros([self.row_batch_size, num_ref_samples], dtype=np.float32)
        batch_predictions = np.zeros([num_eval_samples, self.num_nhoods], dtype=np.int32)
        max_realism_score = np.zeros([num_eval_samples,], dtype=np.float32)
        nearest_indices = np.zeros([num_eval_samples,], dtype=np.int32)

        for begin1 in range(0, num_eval_samples, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_eval_samples)
            feature_batch = eval_features[begin1:end1]

            for begin2 in range(0, num_ref_samples, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_ref_samples)
                ref_batch = self._ref_features[begin2:end2]
                distance_batch[0:end1-begin1, begin2:end2] = self._distance_block.pairwise_distances(feature_batch, ref_batch)

            samples_in_manifold = distance_batch[0:end1-begin1, :, None] <= self.D
            batch_predictions[begin1:end1] = np.any(samples_in_manifold, axis=1).astype(np.int32)

            max_realism_score[begin1:end1] = np.max(self.D[:, 0] / (distance_batch[0:end1-begin1, :] + self.eps), axis=1)
            nearest_indices[begin1:end1] = np.argmin(distance_batch[0:end1-begin1, :], axis=1)

        if return_realism and return_neighbors:
            return batch_predictions, max_realism_score, nearest_indices
        elif return_realism:
            return batch_predictions, max_realism_score
        elif return_neighbors:
            return batch_predictions, nearest_indices

        return batch_predictions


def knn_precision_recall_features(ref_features, eval_features, nhood_sizes=[3],
                                  row_batch_size=10000, col_batch_size=50000):
    """
    k-NN precision and recall of eval_features w.r.t. ref_features.

    Args:
        ref_features: [N, D] reference (real) feature vectors.
        eval_features: [M, D] generated feature vectors.
        nhood_sizes (list): neighbourhood sizes k used to estimate the manifolds.
        row_batch_size, col_batch_size: batch sizes for the pairwise distances.

    Returns:
        dict with 'precision' and 'recall', each of shape [len(nhood_sizes)].
    """
    state = dict()
    num_samples = ref_features.shape[0]

    distance_block = DistanceBlock()
    ref_manifold  = ManifoldEstimator(distance_block, ref_features, row_batch_size, col_batch_size, nhood_sizes)
    eval_manifold = ManifoldEstimator(distance_block, eval_features, row_batch_size, col_batch_size, nhood_sizes)

    print('Evaluating k-NN precision and recall with %i samples...' % num_samples)

    # Precision: fraction of generated samples inside the reference manifold.
    precision = ref_manifold.evaluate(eval_features)
    state['precision'] = precision.mean(axis=0)

    # Recall: fraction of reference samples inside the generated manifold.
    recall = eval_manifold.evaluate(ref_features)
    state['recall'] = recall.mean(axis=0)

    return state
