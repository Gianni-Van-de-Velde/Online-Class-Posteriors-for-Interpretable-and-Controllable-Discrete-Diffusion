"""
Transition kernels, denoising step and classifier-free guidance shared by all samplers.

Adapted from https://github.com/RobinKa/discrete-flow-matching-pytorch
         and https://github.com/lebellig/discrete-fm
"""
import torch

from ..utils import to_one_hot, get_masked_pixels
from ..noise_scheduler import get_scheduler_images as get_scheduler


def init_x(x_in, diffusion_type, vocab_size, batch_size, C, H, W, device):
    """x_in on device, or a fully noised [B, C, H, W] start (uniform tokens / all-mask) if None."""
    if x_in is None:
        if diffusion_type == 'uniform':
            x_in = torch.randint(0, vocab_size, (batch_size, C, H, W))
        else:
            x_in = (vocab_size - 1) * torch.ones((batch_size, C, H, W), dtype=torch.long)
    return x_in.to(device)


def mix_scores(logits_cond, logits_uncond, guid_scale=1):
    """
    Logit-space classifier-free guidance ("Simple Guidance Mechanisms for Discrete Diffusion
    Models", arXiv:2412.10193): softmax(logits_uncond + guid_scale * (logits_cond - logits_uncond)).
    Without logits_uncond, simply softmax(logits_cond).
    """
    if logits_uncond is None:
        return torch.softmax(logits_cond, dim=-1)
    logits = logits_cond + (guid_scale - 1) * (logits_cond - logits_uncond)
    return torch.softmax(logits, dim=-1)


def uniform_ancestral_distr(x_pred, x_one_hot, kappa_t, kappa_t_next, vocab_size):
    """MD4 ancestral (CTMC) kernel q(x_next | x_t, x_pred) for uniform noise, [B, C, H, W, V]."""
    prob_curr = (x_one_hot * x_pred).sum(-1, keepdim=True)
    w_stay    = kappa_t * prob_curr + kappa_t * (1 - kappa_t_next) / (vocab_size * kappa_t_next)
    w_mu      = (kappa_t_next - kappa_t) / vocab_size
    w_uniform = (kappa_t_next - kappa_t) * (1 - kappa_t_next) / (vocab_size**2 * kappa_t_next)
    Z         = kappa_t * prob_curr + (1 - kappa_t) / vocab_size
    return (w_stay * x_one_hot + w_mu * x_pred + w_uniform) / Z


def compute_transition_distr(x_pred, x, kappa_t, kappa_t_next, vocab_size, diffusion_type,
                             use_ancestral_sampling_probs=False):
    """
    Transition distribution q(x_next | x_t, x_pred), [B, C, H, W, V], used to score sampled steps.
    DFM-ODE kernel by default; the MD4 ancestral kernel if use_ancestral_sampling_probs (uniform only).
    """
    x_one_hot = to_one_hot(x, vocab_size)
    if diffusion_type == 'uniform' and use_ancestral_sampling_probs:
        return uniform_ancestral_distr(x_pred, x_one_hot, kappa_t, kappa_t_next, vocab_size)
    coef = (kappa_t_next - kappa_t) / (1.0 - kappa_t).clamp_min(1e-8)
    if diffusion_type == 'masking':
        # Revealed pixels never change, so they have probability 1 under every model. Otherwise
        # the last step (where coef = 1) would rescore the whole image against x_pred.
        coef = coef * get_masked_pixels(x, vocab_size).unsqueeze(-1)
    return x_one_hot + coef * (x_pred - x_one_hot)


def gather_token_transition_logprobs(distribution, sampled_tokens, eps=1e-12):
    """log distribution[sampled_tokens]: [B, C, H, W, V] and [B, C, H, W] -> [B, C, H, W]."""
    probs = torch.gather(distribution, dim=-1, index=sampled_tokens.unsqueeze(-1)).squeeze(-1)
    return torch.log(probs.clamp_min(eps))


def denoising_step(x_pred, x, t, t_next, is_masked, vocab_size, diffusion_type,
                   scheduler_type='linear', polynomial_power=None, use_ancestral_sampling=False):
    """
    One sampling step t -> t_next given the denoiser prediction x_pred [B, C, H, W, V].
    DFM-ODE (Euler) step by default, or the MD4 ancestral step for uniform noise.
    Under masking only the masked pixels (is_masked) can change.

    Returns:
        x_next              : [B, C, H, W]
        transition_logprobs : [B, C, H, W] log-prob of each sampled token under the DFM-ODE kernel
    """
    kappa_t,      _ = get_scheduler(t,      scheduler_type=scheduler_type, polynomial_power=polynomial_power)
    kappa_t_next, _ = get_scheduler(t_next, scheduler_type=scheduler_type, polynomial_power=polynomial_power)

    B, C, H, W, _ = x_pred.size()
    x_one_hot = to_one_hot(x, vocab_size)

    if use_ancestral_sampling and diffusion_type == 'uniform':
        x_next_distr = uniform_ancestral_distr(x_pred, x_one_hot, kappa_t, kappa_t_next, vocab_size)
    else:
        x_next_distr = x_one_hot + (kappa_t_next - kappa_t) / (1.0 - kappa_t) * (x_pred - x_one_hot)

    x_sampled = torch.multinomial(x_next_distr.reshape(-1, vocab_size) + 1e-8, num_samples=1).reshape(B, C, H, W)
    x_next = torch.where(is_masked, x_sampled, x) if diffusion_type == 'masking' else x_sampled

    scoring_distr = compute_transition_distr(x_pred, x, kappa_t, kappa_t_next, vocab_size, diffusion_type)
    transition_logprobs = gather_token_transition_logprobs(scoring_distr, x_sampled)
    return x_next, transition_logprobs
