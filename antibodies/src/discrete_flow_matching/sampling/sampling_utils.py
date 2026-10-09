"""
Transition kernels, denoising step and classifier-free guidance shared by all samplers.

Adapted from https://github.com/RobinKa/discrete-flow-matching-pytorch
         and https://github.com/lebellig/discrete-fm
"""
import torch

from ..utils import to_one_hot
from ..noise_scheduler import get_scheduler


def init_x(x_in, tokenizer, diffusion_type, batch_size, length, vocab_size, device):
    """x_in on device, or a fully noised [B, L] start (uniform tokens / all-mask) if None."""
    if x_in is not None:
        return x_in.to(device)
    if diffusion_type == 'uniform':
        x_in = torch.randint(0, vocab_size, (batch_size, length))
    else:
        x_in = tokenizer.mask_token_id * torch.ones((batch_size, length)).long()
    return x_in.to(device)


def additive_pad_mask(x, tokenizer):
    """[B, L] additive attention mask: 0 on real tokens, -inf on <pad> (what AMPLIFY expects)."""
    attention_mask = torch.zeros_like(x, dtype=torch.float32)
    attention_mask[x == tokenizer.pad_token_id] = float("-inf")
    return attention_mask


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
    """MD4 ancestral (CTMC) kernel q(x_next | x_t, x_pred) for uniform noise, [B, L, V]."""
    prob_curr = (x_one_hot * x_pred).sum(-1, keepdim=True)
    w_stay    = kappa_t * prob_curr + kappa_t * (1 - kappa_t_next) / (vocab_size * kappa_t_next)
    w_mu      = (kappa_t_next - kappa_t) / vocab_size
    w_uniform = (kappa_t_next - kappa_t) * (1 - kappa_t_next) / (vocab_size**2 * kappa_t_next)
    Z         = kappa_t * prob_curr + (1 - kappa_t) / vocab_size
    return (w_stay * x_one_hot + w_mu * x_pred + w_uniform) / Z


def compute_transition_distr(x_pred, x, kappa_t, kappa_t_next, vocab_size, diffusion_type,
                             is_not_special=None, is_masked=None,
                             use_ancestral_sampling_probs=False):
    """
    Transition distribution q(x_next | x_t, x_pred), [B, L, V], used to score sampled steps.
    DFM-ODE kernel by default; the MD4 ancestral kernel if use_ancestral_sampling_probs (uniform only).
    """
    x_one_hot = to_one_hot(x, vocab_size)
    if diffusion_type == 'uniform' and use_ancestral_sampling_probs:
        distr = uniform_ancestral_distr(x_pred, x_one_hot, kappa_t, kappa_t_next, vocab_size)
    else:
        coef = (kappa_t_next - kappa_t) / (1.0 - kappa_t).clamp_min(1e-8)
        distr = x_one_hot + coef * (x_pred - x_one_hot)
    return _protect_special_positions(distr, x_one_hot, diffusion_type, is_not_special, is_masked)


def _protect_special_positions(distr, x_one_hot, diffusion_type, is_not_special, is_masked):
    """
    Positions the generative denoising_step never resamples (special tokens under 'uniform',
    already-revealed tokens under 'masking') provably stay themselves, so force
    distr == x_one_hot there: scoring them then contributes exactly log(1) = 0, whatever
    x_pred said. A position that can never change carries no evidence about which class
    produced the sequence.

    This has no counterpart in the image experiments: images have no special tokens, so there
    the equivalent guard is needed for masking only.
    """
    if diffusion_type == 'uniform' and is_not_special is not None:
        keep = is_not_special.unsqueeze(-1)
    elif diffusion_type == 'masking' and is_masked is not None:
        keep = is_masked.unsqueeze(-1)
    else:
        return distr
    return torch.where(keep, distr, x_one_hot)


def gather_token_transition_logprobs(distribution, sampled_tokens, eps=1e-12):
    """log distribution[sampled_tokens]: [B, L, V] and [B, L] -> [B, L]."""
    probs = torch.gather(distribution, dim=-1, index=sampled_tokens.unsqueeze(-1)).squeeze(-1)
    return torch.log(probs.clamp_min(eps))


def denoising_step(x_pred, x, t, t_next, is_not_special, is_masked, special_token_ids,
                   mask_token_id, diffusion_type, scheduler_type='linear', polynomial_power=1,
                   use_ancestral_sampling=False):
    """
    One sampling step t -> t_next given the denoiser prediction x_pred [B, L, V].
    DFM-ODE (Euler) step by default, or the MD4 ancestral step for uniform noise.

    <pad>/<bos>/<eos> are zeroed out of the sampling distribution so they can never be drawn
    (<mask> stays drawable under masking), and positions that may not change are carried over
    from x: under 'uniform' only non-special positions are resampled, under 'masking' only
    masked ones.

    Returns x_next [B, L]. (The image counterpart also returns the per-token transition
    log-probabilities, which its `sample` records; here no caller uses them, and the samplers
    that need scoring call compute_transition_distr themselves.)
    """
    kappa_t, kappa_t_dot = get_scheduler(t, scheduler_type=scheduler_type, polynomial_power=polynomial_power)
    kappa_t_next, _ = get_scheduler(t_next, scheduler_type=scheduler_type, polynomial_power=polynomial_power)

    vocab_size = x_pred.size(-1)
    x_one_hot = to_one_hot(x, vocab_size)

    if use_ancestral_sampling and diffusion_type == 'uniform':
        x_next_distr = uniform_ancestral_distr(x_pred, x_one_hot, kappa_t, kappa_t_next, vocab_size)
    else:
        coef_t = kappa_t_dot / (1.0 - kappa_t)
        u_t = coef_t * (x_pred - x_one_hot)
        h = t_next - t
        x_next_distr = x_one_hot + h * u_t

    # Hard-mask the special tokens that must never be sampled
    allowed_special_ids = [mask_token_id] if diffusion_type == 'masking' else []
    effective_special_ids = [s for s in special_token_ids if s not in allowed_special_ids]
    mask = torch.ones(x_next_distr.size(-1), device=x_next_distr.device)
    for s_id in effective_special_ids:
        mask[s_id] = 0.0
    mask = mask.view(1, 1, -1).expand_as(x_next_distr)
    x_next_distr = x_next_distr * mask

    batch_size, seq_length, vocab_size = x_next_distr.shape
    # +1e-8 keeps multinomial valid on rows that the hard-mask (or a one-hot x_pred) zeroed out
    x_sampled = torch.multinomial(x_next_distr.reshape(-1, vocab_size) + 1e-8, num_samples=1).view(batch_size, seq_length)

    if diffusion_type == 'uniform':
        x_next = torch.where(is_not_special, x_sampled, x)
    if diffusion_type == 'masking':
        x_next = torch.where(is_masked, x_sampled, x)
    return x_next
