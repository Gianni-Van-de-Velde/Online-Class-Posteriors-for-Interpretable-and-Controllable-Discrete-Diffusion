import os
import torch
import safetensors.torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb

from vision_model.model.dit_model import DiT

# Helper to generate common plot boilerplate
def start_plot(title, ylabel, yscale='linear'):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.xlabel('t')
    plt.ylabel(ylabel)
    plt.yscale(yscale)

def finish_plot(filename, output_fig_dir, suffix):
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{output_fig_dir}/{filename}{suffix}.png")
    plt.close()

def tokens_to_rgb(token_tensor, vocab_size):
    """
    Converts pixel tokens to a displayable uint8 array: [1, H, W] -> [H, W],
    [3, H, W] -> [H, W, 3], and [H, W] -> [H, W, 3]. Tokens are pixel values 0-255; the MASK token (256)
    is clipped to 255. vocab_size is unused, kept for a uniform call signature.
    """
    # Standard 8-bit discrete space handling (pixels mapped 0-255)
    img_np = token_tensor.cpu().numpy()
    if img_np.ndim == 2:  # Grayscale fallback
        img_np = np.stack([img_np]*3, axis=-1)
    elif img_np.ndim == 3 and img_np.shape[0] in [1, 3]:  # [C, H, W] -> [H, W, C]
        img_np = np.transpose(img_np, (1, 2, 0))
        if img_np.shape[2] == 1:  # grayscale: [H, W, 1] -> [H, W]
            img_np = img_np[:, :, 0]
        
    # Keep pixel outputs within bounding ranges
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    return img_np

MASK_DISPLAY_GREY = 128

def tokens_to_display(token_tensor, vocab_size):
    """
    Like tokens_to_rgb, but for figures: always an [H, W, 3] uint8 array, so that imshow shows the
    pixel values in black and white (no colormap, no per-image rescaling). Under masking
    (vocab_size > 256) the MASK token is shown in grey instead of being clipped to white.
    """
    img_np = tokens_to_rgb(token_tensor, vocab_size)
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    if vocab_size > 256:
        tokens = token_tensor.cpu().numpy()
        if tokens.ndim == 3:  # [C, H, W] -> [H, W]: a pixel is masked if any of its channels is
            tokens = tokens.max(axis=0)
        img_np = img_np.copy()
        img_np[tokens == vocab_size - 1] = MASK_DISPLAY_GREY
    return img_np

def plot_and_print_metrics_samples_images(results, xs, final_xs, c, vocab_size, output_fig_dir, checkpoint_number, sample_number,
                                                 save_images=True, save_trajectories=True, save_num_switches_plot=True, add_fig_label=None,
                                                 class_names=None):
    if class_names is None:
        class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'uncond']  # MNIST
    suffix = f"_{add_fig_label}" if add_fig_label else ""
    num_steps, B, C, H, W = xs.size()
    os.makedirs(output_fig_dir, exist_ok=True)
    # =========================================================================
    # Extension A: Save 4 Sample Trajectories across 8 Steps
    # =========================================================================
    if save_trajectories and xs.ndim == 5:

        # Choose 4 samples (or fewer if the batch size is smaller)
        num_trajectory_images = min(4, B)
        # Select 8 evenly spaced steps from the generative timeline
        step_indices = np.linspace(0, num_steps - 1, 8).astype(int)
        
        for idx in range(num_trajectory_images):
            fig, axes = plt.subplots(1, 8, figsize=(16, 2.5))
            class_label = class_names[c[idx].item()] if c is not None else str(idx)
            fig.suptitle(f"Trajectory Evolution Sample {idx}: Class {class_label} (Checkpoint {checkpoint_number})", y=1.05)
            
            for step_col, step_idx in enumerate(step_indices):
                img_rgb = tokens_to_display(xs[step_idx, idx], vocab_size)
                axes[step_col].imshow(img_rgb)
                axes[step_col].set_title(f"Step {step_idx}", fontsize=9)
                axes[step_col].axis('off')
                
            plt.tight_layout()
            plt.savefig(f"{output_fig_dir}/ckpt_{checkpoint_number}_generation_sample_{idx}_trajectory{suffix}.png", bbox_inches='tight')
            plt.close()

    # =========================================================================
    # Extension B: Save Final Generated Samples in a Grid Layout
    # =========================================================================
    if save_images and final_xs.ndim == 4:        
        grid_size = 4 if B >= 16 else 2
        total_grid_cells = grid_size * grid_size
        samples_to_plot = min(total_grid_cells, B)
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2.5, grid_size * 2.5 + 0.5))
        fig.suptitle(f"Final Generated Samples Grid {grid_size}x{grid_size} (Checkpoint {checkpoint_number})", y=1.02)
        
        # Flatten axes array for simple iteration mapping
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else np.array([axes])
        
        for i in range(total_grid_cells):
            if i < samples_to_plot:
                img_rgb = tokens_to_display(final_xs[i], vocab_size)
                axes_flat[i].imshow(img_rgb)
                
                # Retrieve label index string if c is provided
                if c is not None:
                    class_idx = int(c[i].item())
                    # Ensure safety if dropout dropped conditions used alternative index values (-1 or out-of-bounds)
                    if 0 <= class_idx < len(class_names) - 1:
                        class_text = class_names[class_idx]
                    else:
                        class_text = class_names[-1]  # last entry is always 'uncond'
                    
                    # Add text below the sub-image
                    axes_flat[i].set_title(class_text, y=-0.15, fontsize=10)
                    
            axes_flat[i].axis('off')
            
        plt.tight_layout()
        plt.savefig(f"{output_fig_dir}/ckpt_{checkpoint_number}_generation_final_samples_{grid_size}x{grid_size}_grid{suffix}.png", bbox_inches='tight')
        plt.close()

    # =========================================================================
    # Extension C: Save FSwitches analysis
    # =========================================================================
    ts = torch.linspace(0,1,num_steps-1)

    if save_num_switches_plot:
        avg_num_per_step = C*H*W/num_steps
        start_plot("Number of token switches per step", "Number of token switches", yscale="linear")
        plt.plot(ts, results["num_switches_per_step"], 'c-',label='Model')
        plt.plot(ts, avg_num_per_step*torch.ones(num_steps-1), 'b--',label=r'Expected ($\frac{C H W}{T}$)')
        finish_plot(f"ckpt_{checkpoint_number}_generation_num_switches", output_fig_dir, suffix)

        total_number = C*H*W
        start_plot("Cummulative number of token switches", "Number of token switches", yscale="linear")
        plt.plot(ts, results["total_trajectory_switches"], 'c-',label='Model')
        plt.plot(ts, total_number*torch.ones(num_steps-1), 'b--',label=r'Expected ($C H W$)')
        finish_plot(f"ckpt_{checkpoint_number}_generation_num_switches_cum", output_fig_dir, suffix)


# ─────────────────────────────────────────────────────────────────────────────
# Posterior/classifier agreement plots (posterior_classifier_agreement.py)
# ─────────────────────────────────────────────────────────────────────────────

PAPER_RC_PARAMS = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
}

LEGEND_KWARGS = dict(
    ncol=2, fontsize=10.5,
    handlelength=1.0, handletextpad=0.4,
    columnspacing=0.8, labelspacing=0.25, borderpad=0.3,
)


def get_class_spectrum_colors(num_classes):
    """Distinct, evenly-spaced colors spanning red (class 0) -> blue (last class)."""
    hues = np.linspace(0.0, 0.667, num_classes)
    return [hsv_to_rgb((h, 0.9, 0.9)) for h in hues]


def _save_fig(fig, path, dpi=150):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_agreement_vs_time(t, agreement, num_classes, title, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t, agreement, marker="o", color="tab:blue")
    ax.axhline(1.0 / num_classes, color="gray", linestyle="--", linewidth=0.8, label="chance level")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"Diffusion time $t$")
    ax.set_ylabel("Agreement with ViT classifier")
    ax.set_title(title, fontsize=12)
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
    ax.set_xlabel(r"Diffusion time $t$")
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
    ax_nll.set_ylabel("NLL of online class posterior\nat ViT-predicted class")
    ax_nll.set_title(title)
    ax_nll.legend(fontsize=9)
    ax_nll.grid(alpha=0.4)

    ax_agree.plot(temps, agreement, marker="o", color="tab:blue")
    ax_agree.set_xlabel(r"temp")
    ax_agree.set_ylabel("Agreement with\nViT classifier")
    ax_agree.set_ylim(0, 1)
    ax_agree.grid(alpha=0.4)
    _save_fig(fig, path)


def plot_roc_curves(curves, title, path):
    """curves: list of (fpr, tpr, label, color)."""
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.8, label="chance")
    for fpr, tpr, label, color in curves:
        ax.plot(fpr, tpr, color=color, linewidth=2, label=label)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.4)
    ax.set_aspect("equal")
    _save_fig(fig, path)


def plot_confusion_matrix(confusion, tick_labels, title, path):
    """confusion[i, j]: samples the ViT calls i and the online class posterior calls j."""
    num_classes = len(tick_labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xlabel("Online class posterior prediction")
    ax.set_ylabel("ViT classifier prediction")
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


def plot_class_distribution(dfm_pct, vit_pct, tick_labels, title, path):
    num_classes = len(tick_labels)
    bar_x = np.arange(num_classes)
    bar_width = 0.38

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(bar_x - bar_width / 2, dfm_pct, bar_width, label="Online class posterior", color="tab:blue")
    ax.bar(bar_x + bar_width / 2, vit_pct, bar_width, label="ViT classifier", color="tab:orange")
    ax.axhline(100.0 / num_classes, color="gray", linestyle="--", linewidth=1.0, label="uniform (1/K)")
    ax.set_xticks(bar_x)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_ylabel("Share of samples (%)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.4, axis="y")
    _save_fig(fig, path)


def plot_posterior_trajectory(t, probs, final_rgb, labels, colors, frame_color, title, ylabel, path,
                              legend_above=False):
    """probs: [T, K] posterior of one trajectory; the final image is drawn as an inset."""
    num_classes = len(labels)
    fig, ax = plt.subplots(figsize=(8, 4))
    for k in range(num_classes):
        ax.plot(t, probs[:, k], label=labels[k], color=colors[k])
    ax.axhline(1.0 / num_classes, color='gray', linestyle='--', linewidth=0.8, label='prior')
    ax.set_title(title)
    ax.set_xlabel('Diffusion time t')
    ax.set_ylabel(ylabel)

    if legend_above:
        # Headroom above 1.0 holds the legend so it never overlaps the curves.
        y_top = 1.2
        ax.set_ylim(0, y_top)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(alpha=0.4)
        ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.0 / y_top, 1.0, (y_top - 1.0) / y_top),
                  mode='expand', **dict(LEGEND_KWARGS, ncol=num_classes + 1))
    else:
        ax.legend(**LEGEND_KWARGS)
        ax.grid(alpha=0.4)

    inset_ax = ax.inset_axes([0.74, 0.22, 0.24, 0.4])
    inset_ax.imshow(final_rgb)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    for spine in inset_ax.spines.values():
        spine.set_edgecolor(frame_color)
        spine.set_linewidth(3)
    _save_fig(fig, path, dpi=120)


def plot_agreement_vs_margin(bin_centers, agreement, counts, num_classes, title, path):
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
    ax.set_xlabel(r"DFM confidence margin  $p_{(1)}(c|x_t) - p_{(2)}(c|x_t)$")
    ax.set_ylabel("Agreement with ViT classifier")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.4)
    _save_fig(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dit_checkpoint(run_name, checkpoint_number, model_config, device):
    """DiT with the EMA weights of logs/<run>/checkpoints/checkpoint_<k>/ (falls back to the raw weights)."""
    model = DiT(model_config)
    ckpt_dir = f"logs/{run_name}/checkpoints/checkpoint_{checkpoint_number}"
    ckpt_path = f"{ckpt_dir}/ema_model.safetensors"
    if not os.path.exists(ckpt_path):
        ckpt_path = f"{ckpt_dir}/model.safetensors"
    print(f"  [{run_name}] Loading: {ckpt_path}")
    model.load_state_dict(safetensors.torch.load_file(ckpt_path, device="cpu"))
    return model.to(device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# Pretrained classifier wrapper
# ─────────────────────────────────────────────────────────────────────────────

_CLASSIFIER_DEFAULTS = {
    "cifar10": "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10",
    "mnist":   "farleyknight-org-username/vit-base-mnist",
}

# Last entry is the unconditional (null) class token.
DATASET_CLASS_NAMES = {
    'cifar10': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'uncond'],
    'mnist':   ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'uncond'],
}


class _ClassifierWrapper:
    """Thin wrapper around a HuggingFace image-classification model."""

    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
        self.num_classes = model.config.num_labels
        # Label names in the classifier's own index order — don't assume it
        # matches whatever class_names list the caller happens to be using.
        self.class_names = [model.config.id2label[i] for i in range(self.num_classes)]

    @torch.no_grad()
    def predict(self, images, batch_size=64):
        """
        Args:
            images: list of PNG file paths, or already-loaded PIL Images
        Returns:
            class_indices : LongTensor [N]
            confidences   : FloatTensor [N]  (max softmax prob)
            all_probs     : FloatTensor [N, num_classes]
        """
        from PIL import Image as PILImage
        all_probs = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            pil_imgs = [
                PILImage.open(img).convert("RGB") if isinstance(img, (str, os.PathLike)) else img.convert("RGB")
                for img in batch
            ]
            inputs = self.processor(images=pil_imgs, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logits = self.model(**inputs).logits
            all_probs.append(torch.softmax(logits, dim=-1).cpu())
        all_probs = torch.cat(all_probs, dim=0)          # [N, C]
        confidences, class_indices = all_probs.max(dim=-1)
        return class_indices, confidences, all_probs


def get_classifier(dataset_name, device="cpu", model_name=None):
    """
    Returns a pretrained classifier for the given dataset (see _CLASSIFIER_DEFAULTS).

    Usage:
        clf = get_classifier("mnist", device="cuda")
        class_idx, conf, probs = clf.predict(list_of_png_paths)
    """
    from transformers import AutoModelForImageClassification, AutoImageProcessor

    if model_name is None:
        model_name = _CLASSIFIER_DEFAULTS.get(dataset_name)
        if model_name is None:
            raise ValueError(f"No default classifier for '{dataset_name}'. Pass model_name explicitly.")

    print(f"Loading classifier: {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name).to(device).eval()
    print(f"  num_classes={model.config.num_labels}, device={device}")
    return _ClassifierWrapper(model, processor, device)