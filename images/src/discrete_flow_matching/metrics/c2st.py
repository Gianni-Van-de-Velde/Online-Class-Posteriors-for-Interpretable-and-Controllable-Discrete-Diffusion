import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


def compute_c2st(real_feats: torch.Tensor, gen_feats: torch.Tensor,
                 test_size: float = 0.2, random_state: int = 42):
    """
    Classifier Two-Sample Test (C2ST): logistic regression trained to separate real (1) from
    generated (0) features, evaluated on a held-out split.
      - accuracy ~0.5  ->  indistinguishable  ->  good generator
      - accuracy ~1.0  ->  trivially separable ->  bad generator

    Returns (accuracy, log_loss) on the held-out split.
    """
    n = min(real_feats.shape[0], gen_feats.shape[0])
    X = np.concatenate([real_feats[:n].numpy(), gen_feats[:n].numpy()], axis=0)
    y = np.array([1] * n + [0] * n)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    ll = log_loss(y_test, clf.predict_proba(X_test))
    return float(acc), float(ll)
