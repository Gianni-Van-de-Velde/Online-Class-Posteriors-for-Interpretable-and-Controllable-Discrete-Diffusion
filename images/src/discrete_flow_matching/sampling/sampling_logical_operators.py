"""Compositional NOT / OR samplers driven by the online class posterior, and their baselines."""
import math
import torch

from ..utils import get_masked_pixels
from ..noise_scheduler import get_scheduler_images as get_scheduler
from .sampling_utils import (
    init_x,
    mix_scores,
    denoising_step,
    compute_transition_distr,
    gather_token_transition_logprobs,
)
from .sampling_standard import sample


def sample_NOT(cond_model, c, c_uncond,
               guid_scale_max=5.0,
               initial_guid_scale=1.5,
               x_in=None,
               diffusion_type='uniform',
               scheduler_type='linear',
               polynomial_power=None,
               use_ancestral_sampling=False,
               num_steps=100, vocab_size=256,
               batch_size=4, C=3, H=32, W=32,
               temp=1.0,
               return_trajectory=True,
               use_ancestral_sampling_probs=False,
               uncond_model=None):
    """
    Compositional NOT: the unconditional process steered away from class c,

        x_pred_NOT = x_pred_uncond - guid_scale * (x_pred_c - x_pred_uncond),

    clamped to be non-negative and renormalised over the vocabulary. This follows from the
    rate-level NOT gate in the paper, since both denoising steps are affine in x_pred.

    guid_scale is set per sample from the unnormalised posterior of c against the unconditional model:

        log_post  += temp * (log q(x_next | x_t, c) - log q(x_next | x_t, uncond))
        p_c        = min(exp(log_post), p_c_max),   p_c_max = guid_scale_max / (1 + guid_scale_max)
        guid_scale = p_c / (1 - p_c)

    log_post starts at log(initial_guid_scale / (1 + initial_guid_scale)), so the first step uses
    initial_guid_scale (0 = no guidance) rather than the maximum.

    cond_model runs the c-conditioned passes; uncond_model (default: cond_model) runs the
    c_uncond-conditioned ones and therefore drives the base process.

    Returns:
        final_x     : [B, C, H, W]
        xs          : [T, B, C, H, W] trajectory           (None if not return_trajectory)
        post_probs  : [T, B] clamped p_c used at each step (None if not return_trajectory)
        guid_scales : [T, B] guid_scale used at each step  (None if not return_trajectory)
    """
    if uncond_model is None:
        uncond_model = cond_model
    device = cond_model.device
    x = init_x(x_in, diffusion_type, vocab_size, batch_size, C, H, W, device)
    batch_size, C, H, W = x.shape
    t_axis = torch.linspace(0, 1, num_steps)

    if initial_guid_scale < 0:
        raise ValueError(f"initial_guid_scale must be >= 0, got {initial_guid_scale}")
    p_c_init = initial_guid_scale / (1.0 + initial_guid_scale)
    log_post_init = math.log(p_c_init) if p_c_init > 0 else float('-inf')
    log_post = torch.full((batch_size,), log_post_init, device=device)
    p_c_max = guid_scale_max / (1.0 + guid_scale_max)

    xs, post_probs_list, guid_scales_list = [], [], []
    for t, t_next in zip(t_axis[:-1], t_axis[1:]):
        kappa_t,      _ = get_scheduler(t,      scheduler_type=scheduler_type, polynomial_power=polynomial_power)
        kappa_t_next, _ = get_scheduler(t_next, scheduler_type=scheduler_type, polynomial_power=polynomial_power)
        kappa_ts = kappa_t * torch.ones(batch_size, device=device)

        if return_trajectory:
            xs.append(x.clone())

        p_c = torch.exp(log_post).clamp(max=p_c_max)
        guid_scale = p_c / (1.0 - p_c)

        with torch.no_grad():
            logits_c      = cond_model(x, kappa_ts, c)
            logits_uncond = uncond_model(x, kappa_ts, c_uncond)
        x_pred_c      = torch.softmax(logits_c,      dim=-1)
        x_pred_uncond = torch.softmax(logits_uncond, dim=-1)

        gs = guid_scale.view(batch_size, 1, 1, 1, 1)
        x_pred_not = x_pred_uncond - gs * (x_pred_c - x_pred_uncond)
        x_pred_not = x_pred_not.clamp(min=0.0)
        x_pred_not = x_pred_not / x_pred_not.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        is_masked = get_masked_pixels(x, vocab_size)
        x_next, _ = denoising_step(
            x_pred_not, x, t, t_next, is_masked, vocab_size, diffusion_type,
            scheduler_type=scheduler_type, polynomial_power=polynomial_power,
            use_ancestral_sampling=use_ancestral_sampling,
        )

        distr_c      = compute_transition_distr(x_pred_c,      x, kappa_t, kappa_t_next, vocab_size, diffusion_type, use_ancestral_sampling_probs)
        distr_uncond = compute_transition_distr(x_pred_uncond, x, kappa_t, kappa_t_next, vocab_size, diffusion_type, use_ancestral_sampling_probs)
        log_p_c      = gather_token_transition_logprobs(distr_c,      x_next).sum(dim=[1, 2, 3])  # [B]
        log_p_uncond = gather_token_transition_logprobs(distr_uncond, x_next).sum(dim=[1, 2, 3])
        log_post = log_post + temp * (log_p_c - log_p_uncond)

        if return_trajectory:
            post_probs_list.append(p_c.detach().clone())
            guid_scales_list.append(guid_scale.detach().clone())

        x = x_next

    if not return_trajectory:
        return x, None, None, None
    return x, torch.stack(xs), torch.stack(post_probs_list), torch.stack(guid_scales_list)


def sample_NOT_baseline(cond_model, c, c_uncond,
                        lam_constant=1.0,
                        x_in=None,
                        diffusion_type='uniform',
                        scheduler_type='linear',
                        polynomial_power=None,
                        use_ancestral_sampling=False,
                        num_steps=100, vocab_size=256,
                        batch_size=4, C=3, H=32, W=32,
                        return_trajectory=True,
                        uncond_model=None):
    """
    Static NOT baseline: logit-space CFG (mix_scores) with a constant negative scale,

        logits = logits_uncond - lam_constant * (logits_c - logits_uncond).

    Unlike sample_NOT, this extrapolates logits rather than probabilities, with a fixed strength.
    Same cond_model / uncond_model split as sample_NOT.

    Returns:
        final_x : [B, C, H, W]
        xs      : [T, B, C, H, W] trajectory (None if not return_trajectory)
    """
    if uncond_model is None:
        uncond_model = cond_model
    device = cond_model.device
    x = init_x(x_in, diffusion_type, vocab_size, batch_size, C, H, W, device)
    batch_size, C, H, W = x.shape
    t_axis = torch.linspace(0, 1, num_steps)

    xs = []
    for t, t_next in zip(t_axis[:-1], t_axis[1:]):
        kappa_t, _ = get_scheduler(t, scheduler_type=scheduler_type, polynomial_power=polynomial_power)
        kappa_ts = kappa_t * torch.ones(batch_size, device=device)

        if return_trajectory:
            xs.append(x.clone())

        with torch.no_grad():
            logits_c      = cond_model(x, kappa_ts, c)
            logits_uncond = uncond_model(x, kappa_ts, c_uncond)
        x_pred = mix_scores(logits_c, logits_uncond, guid_scale=-lam_constant)

        is_masked = get_masked_pixels(x, vocab_size)
        x, _ = denoising_step(
            x_pred, x, t, t_next, is_masked, vocab_size, diffusion_type,
            scheduler_type=scheduler_type, polynomial_power=polynomial_power,
            use_ancestral_sampling=use_ancestral_sampling,
        )

    if not return_trajectory:
        return x, None
    return x, torch.stack(xs)


def sample_OR(model, c1, c2, c_uncond,
              x_in=None,
              diffusion_type='uniform',
              scheduler_type='linear',
              polynomial_power=None,
              use_ancestral_sampling=False,
              num_steps=100, vocab_size=256,
              batch_size=4, C=3, H=32, W=32,
              temp=1.0,
              return_trajectory=True,
              use_ancestral_sampling_probs=False,
              posterior_type='softmax'):
    """
    Compositional OR of classes c1 and c2: denoise with the posterior-weighted mixture

        x_pred_OR = w1 * x_pred_c1 + w2 * x_pred_c2,

    where log_post (starting at 0) accumulates each class's log-ratio against the unconditional
    model, as in sampler_with_posterior, and
      posterior_type='softmax': w = softmax(log_post), a convex combination;
      posterior_type='unnorm' : w = exp(log_post), unnormalised (w1 + w2 != 1 in general).
    Class predictions are unguided; c_uncond only serves as the scoring baseline.

    Returns:
        final_x    : [B, C, H, W]
        xs         : [T, B, C, H, W] trajectory           (None if not return_trajectory)
        post_probs : [T, B, 2] weights after each step    (None if not return_trajectory)
    """
    device = model.device
    x = init_x(x_in, diffusion_type, vocab_size, batch_size, C, H, W, device)
    batch_size, C, H, W = x.shape
    t_axis = torch.linspace(0, 1, num_steps)

    log_post = torch.zeros((batch_size, 2), device=device)

    xs, post_probs_list = [], []
    for t, t_next in zip(t_axis[:-1], t_axis[1:]):
        kappa_t,      _ = get_scheduler(t,      scheduler_type=scheduler_type, polynomial_power=polynomial_power)
        kappa_t_next, _ = get_scheduler(t_next, scheduler_type=scheduler_type, polynomial_power=polynomial_power)
        kappa_ts = kappa_t * torch.ones(batch_size, device=device)

        if return_trajectory:
            xs.append(x.clone())

        if posterior_type == 'softmax':
            w = torch.softmax(log_post, dim=-1)
        else:
            w = torch.exp(log_post)

        with torch.no_grad():
            logits_c1     = model(x, kappa_ts, c1)
            logits_c2     = model(x, kappa_ts, c2)
            logits_uncond = model(x, kappa_ts, c_uncond)
        x_pred_c1     = torch.softmax(logits_c1,     dim=-1)
        x_pred_c2     = torch.softmax(logits_c2,     dim=-1)
        x_pred_uncond = torch.softmax(logits_uncond, dim=-1)

        w1 = w[:, 0].view(batch_size, 1, 1, 1, 1)
        w2 = w[:, 1].view(batch_size, 1, 1, 1, 1)
        x_pred_or = w1 * x_pred_c1 + w2 * x_pred_c2

        is_masked = get_masked_pixels(x, vocab_size)
        x_next, _ = denoising_step(
            x_pred_or, x, t, t_next, is_masked, vocab_size, diffusion_type,
            scheduler_type=scheduler_type, polynomial_power=polynomial_power,
            use_ancestral_sampling=use_ancestral_sampling,
        )

        distr_c1     = compute_transition_distr(x_pred_c1,     x, kappa_t, kappa_t_next, vocab_size, diffusion_type, use_ancestral_sampling_probs)
        distr_c2     = compute_transition_distr(x_pred_c2,     x, kappa_t, kappa_t_next, vocab_size, diffusion_type, use_ancestral_sampling_probs)
        distr_uncond = compute_transition_distr(x_pred_uncond, x, kappa_t, kappa_t_next, vocab_size, diffusion_type, use_ancestral_sampling_probs)
        log_p_c1     = gather_token_transition_logprobs(distr_c1,     x_next).sum(dim=[1, 2, 3])  # [B]
        log_p_c2     = gather_token_transition_logprobs(distr_c2,     x_next).sum(dim=[1, 2, 3])
        log_p_uncond = gather_token_transition_logprobs(distr_uncond, x_next).sum(dim=[1, 2, 3])

        log_post[:, 0] = log_post[:, 0] + temp * (log_p_c1 - log_p_uncond)
        log_post[:, 1] = log_post[:, 1] + temp * (log_p_c2 - log_p_uncond)

        if return_trajectory:
            if posterior_type == 'softmax':
                post_probs_list.append(torch.softmax(log_post, dim=-1).detach().clone())
            else:
                post_probs_list.append(torch.exp(log_post).detach().clone())

        x = x_next

    if not return_trajectory:
        return x, None, None
    return x, torch.stack(xs), torch.stack(post_probs_list)


def sample_OR_baseline(model, c1, c2, c_uncond,
                       coin_prob=0.5,
                       guid_scale=1.0,
                       x_in=None,
                       diffusion_type='uniform',
                       scheduler_type='linear',
                       polynomial_power=None,
                       use_ancestral_sampling=False,
                       num_steps=100, vocab_size=256,
                       batch_size=4, C=3, H=32, W=32,
                       return_trajectory=True):
    """
    OR baseline: each sample independently picks c1 (with probability coin_prob) or c2 and is then
    generated by standard CFG sampling (`sample`) for that class alone.

    Returns:
        final_x : [B, C, H, W]
        xs      : [T, B, C, H, W] trajectory (None if not return_trajectory)
        coin    : [B] bool, True where the sample was generated under c1
    """
    batch_size = c1.shape[0]
    coin = torch.bernoulli(torch.full((batch_size,), coin_prob, device=c1.device)).bool()
    c_mixed = torch.where(coin, c1, c2)

    x, xs, _, _ = sample(
        model, c_mixed, c_uncond=c_uncond, guid_scale=guid_scale, x_in=x_in,
        diffusion_type=diffusion_type, scheduler_type=scheduler_type,
        polynomial_power=polynomial_power, use_ancestral_sampling=use_ancestral_sampling,
        num_steps=num_steps, vocab_size=vocab_size,
        batch_size=batch_size, C=C, H=H, W=W,
        return_trajectory=return_trajectory,
    )
    return x, xs, coin
