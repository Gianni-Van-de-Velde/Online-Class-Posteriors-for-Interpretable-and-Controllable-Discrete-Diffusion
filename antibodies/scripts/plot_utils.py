"""
Plotting helpers for the posterior diagnostics written by posterior_classifier_agreement.py.

The same eight figures as the image experiments' counterpart, so the two can be read side by
side -- see ../Image_experiments/scripts/image_utils.py. The ground truth here is real IgBLAST
rather than a pretrained ViT, and the class distribution carries an extra reference line,
because real V-gene families are naturally imbalanced where MNIST/CIFAR classes are not.

These are diagnostics, written next to the data as PNGs. The paper's own figures are not built
here: the numbers behind them are the CSVs under logs/Results/ and logs/Figures/.
"""
import matplotlib.pyplot as plt
import numpy as np


def _save_fig(fig, path, dpi=150, bbox_inches=None):
    """Writes one diagnostic PNG at an explicit path and closes the figure."""
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)


def plot_agreement_vs_time(t, agreement, num_classes, title, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t, agreement, marker="o", color="tab:blue")
    ax.axhline(1.0 / num_classes, color="gray", linestyle="--", linewidth=0.8, label="chance level")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("Agreement with IgBLAST")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.4)
    _save_fig(fig, path)


def plot_auc_vs_time(t, auc_per_class, auc_macro, valid_classes, ylabel, title, path):
    """auc_per_class: [T, K]; only the columns in valid_classes are drawn."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in valid_classes:
        ax.plot(t, auc_per_class[:, c], color="tab:blue", alpha=0.15, linewidth=0.8)
    ax.plot(t, auc_macro, color="tab:blue", linewidth=2, label="macro one-vs-rest AUC")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance level")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(f"One-vs-rest AUC ({ylabel})")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.4)
    _save_fig(fig, path)


def plot_temp_sweep(temps, nll, agreement, best_temp, title, path):
    fig, (ax_nll, ax_agree) = plt.subplots(2, 1, figsize=(6, 6.5), sharex=True)
    ax_nll.plot(temps, nll, marker="o", color="tab:blue")
    ax_nll.axvline(best_temp, color="gray", linestyle="--", linewidth=0.8,
                   label=f"best temp={best_temp:.3f}")
    ax_nll.set_ylabel("NLL of online class posterior\nat IgBLAST-called class")
    ax_nll.set_title(title)
    ax_nll.legend(fontsize=9)
    ax_nll.grid(alpha=0.4)

    ax_agree.plot(temps, agreement, marker="o", color="tab:blue")
    ax_agree.set_xlabel("temp")
    ax_agree.set_ylabel("Agreement with\nIgBLAST")
    ax_agree.set_ylim(0, 1)
    ax_agree.grid(alpha=0.4)
    _save_fig(fig, path)


def plot_roc_curves(curves, title, path):
    """curves: list of (fpr, tpr, label, color); color=None takes the next cycle colour."""
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.8, label="chance")
    for fpr, tpr, label, color in curves:
        ax.plot(fpr, tpr, color=color, linewidth=2, label=label)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.4)
    ax.set_aspect("equal")
    _save_fig(fig, path)


def plot_confusion_matrix(confusion, tick_labels, title, path):
    """confusion[i, j]: samples IgBLAST calls i and the online class posterior calls j."""
    num_classes = len(tick_labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xlabel("Online class posterior prediction")
    ax.set_ylabel("IgBLAST prediction")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels)
    ax.set_title(title)
    thresh = confusion.max() / 2 if confusion.max() > 0 else 1
    for yi in range(num_classes):
        for xi in range(num_classes):
            ax.text(xi, yi, confusion[yi, xi], ha="center", va="center",
                    color="white" if confusion[yi, xi] > thresh else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="count")
    _save_fig(fig, path)


def plot_class_distribution(tracked_pct, igblast_pct, empirical_pct, tick_labels, title, path):
    """Unlike the image experiments' balanced classes, V-gene families are naturally
    imbalanced, so the reference pool's own empirical frequency is drawn as the honest
    reference line. Uniform 1/K is kept for context."""
    num_classes = len(tick_labels)
    bar_x = np.arange(num_classes)
    bar_width = 0.38

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(bar_x - bar_width / 2, tracked_pct, bar_width, label="Online class posterior", color="tab:blue")
    ax.bar(bar_x + bar_width / 2, igblast_pct, bar_width, label="IgBLAST", color="tab:orange")
    ax.plot(bar_x, empirical_pct, color="black", marker="D", markersize=5, linewidth=1.2,
            linestyle="-", label="reference-pool empirical freq.")
    ax.axhline(100.0 / num_classes, color="gray", linestyle="--", linewidth=1.0, label="uniform (1/K)")
    ax.set_xticks(bar_x)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_ylabel("Share of samples (%)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.4, axis="y")
    _save_fig(fig, path)


def plot_posterior_trajectory(t, probs, class_names, title, ylabel, path, log_scale=False):
    """probs: [T, K] for one trajectory.

    log_scale=False -- the softmax posterior (sums to 1 at every t, what the online
        prediction's argmax is taken from): linear y-axis, fixed [0, 1.05].
    log_scale=True  -- the raw exp(accumulated log-likelihood) BEFORE the softmax, which does
        not sum to 1: each class's evidence considered on its own, against nothing but the
        flat prior it started from. That quantity is exp(cumsum(log-ratios)), so a class's
        curve is exponential in the number of steps of evidence for or against it, and a log
        axis is what makes it legible -- on a linear one, everything but the eventual winner
        is indistinguishable from the x-axis almost immediately.
    """
    num_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = plt.cm.tab10.colors
    for k, name in enumerate(class_names):
        ax.plot(t, probs[:, k], label=name, color=colors[k % len(colors)])
    ax.axhline(1.0 / num_classes, color='gray', linestyle='--', linewidth=0.8, label='prior')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(alpha=0.4)
    _save_fig(fig, path, dpi=130, bbox_inches="tight")


def plot_agreement_vs_margin(bin_centers, agreement, counts, num_classes, title, path):
    """Empty bins are the caller's to filter out -- `counts` sets the marker areas."""
    sizes = 20 + 80 * (counts / counts.max())
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(bin_centers, agreement, s=sizes, color="tab:blue", zorder=3)
    for x, y, n in zip(bin_centers, agreement, counts):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8)
    ax.plot([0, 1], [1.0 / num_classes, 1.0 / num_classes], color="gray", linestyle="--",
            linewidth=0.8, label="chance level")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"Online class posterior margin  $p_{(1)}(c|x_t) - p_{(2)}(c|x_t)$")
    ax.set_ylabel("Agreement with IgBLAST")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.4)
    _save_fig(fig, path)
