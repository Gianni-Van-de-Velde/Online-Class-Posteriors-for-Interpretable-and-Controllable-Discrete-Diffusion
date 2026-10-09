"""Class-conditional sampling with classifier-free guidance, optionally tracking the class posterior."""
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


def sample(model, c, c_uncond=None, guid_scale=1, x_in=None,
           diffusion_type='uniform', scheduler_type='linear', polynomial_power=None,
           use_ancestral_sampling=False, num_steps=100, vocab_size=256,
           batch_size=4, C=3, H=32, W=32, return_trajectory=True):
    """
    Class-conditional sampling with logit-space CFG (see mix_scores); no guidance if c_uncond is None.

    Returns:
        final_x        : [B, C, H, W]
        xs             : [T, B, C, H, W]    trajectory           (None if not return_trajectory)
        x_preds        : [T, B, C, H, W, V] guided predictions   (None if not return_trajectory)
        trans_logprobs : [T, B, C, H, W]    log-prob of each sampled token under the DFM-ODE kernel
                                            (None if not return_trajectory)
    """
    x = init_x(x_in, diffusion_type, vocab_size, batch_size, C, H, W, model.device)
    batch_size, C, H, W = x.shape
    t_axis = torch.linspace(0, 1, num_steps)

    xs, x_preds, trans_logprobs = [], [], []
    for t, t_next in zip(t_axis[:-1], t_axis[1:]):
        # The model is conditioned on the noise level kappa(t), not on t itself.
        kappa_t, _ = get_scheduler(t, scheduler_type=scheduler_type, polynomial_power=polynomial_power)
        kappa_ts = kappa_t * torch.ones(batch_size, device=model.device)

        with torch.no_grad():
            logits_cond = model(x, kappa_ts, c)
            logits_uncond = model(x, kappa_ts, c_uncond) if c_uncond is not None else None
        x_pred = mix_scores(logits_cond, logits_uncond, guid_scale=guid_scale)

        is_masked = get_masked_pixels(x, vocab_size)
        if return_trajectory:
            xs.append(x)
            x_preds.append(x_pred)
        x, transition_logprobs = denoising_step(
            x_pred, x, t, t_next, is_masked, vocab_size, diffusion_type,
            scheduler_type=scheduler_type, polynomial_power=polynomial_power,
            use_ancestral_sampling=use_ancestral_sampling,
        )
        if return_trajectory:
            trans_logprobs.append(transition_logprobs)

    if not return_trajectory:
        return x, None, None, None
    return x, torch.stack(xs), torch.stack(x_preds), torch.stack(trans_logprobs)


def sampler_with_posterior(cond_model, c, c_uncond, num_classes,
                           guid_scale=1, x_in=None,
                           diffusion_type='uniform', scheduler_type='linear', polynomial_power=None,
                           use_ancestral_sampling=False,
                           num_steps=100, vocab_size=256,
                           batch_size=4, C=3, H=32, W=32, temp=0.1, offset=0.0,
                           uncond_model=None,
                           use_ancestral_sampling_probs=False,
                           normalize_by_unmask_count=False):
    """
    CFG sampling (as in `sample`) while tracking the class posterior p(k | x_t) for all K classes.

    Each sampled transition x_t -> x_next is scored under every class-conditional model and under
    the unconditional model, and the log-ratios are accumulated from the uniform prior log(1/K):

        log_post_k += temp * (log q(x_next | x_t, k) - log q(x_next | x_t, uncond)) + offset

    The scoring kernel is the DFM-ODE kernel, or the MD4 ancestral kernel if
    use_ancestral_sampling_probs (independent of use_ancestral_sampling, which only sets the
    generative step). cond_model runs every class-conditioned forward pass and uncond_model
    (default: cond_model) every c_uncond-conditioned one, for CFG and for the posterior alike.

    normalize_by_unmask_count (masking only): divide each step's log-ratio by the number of pixels
    unmasked in that step (at least 1), so that steps revealing many pixels do not dominate.

    Returns:
        final_x           : [B, C, H, W]
        xs                : [T, B, C, H, W] trajectory (T = num_steps - 1)
        post_probs        : [T, B, K] softmax(log_post) after each step
        unnorm_post_probs : [T, B, K] exp(log_post) after each step
        raw_logratios     : [T, B, K] per-step log-ratios before temp and offset, so the posterior
                            can be recomputed for any temp without re-sampling
    """
    if uncond_model is None:
        uncond_model = cond_model
    device = cond_model.device
    x = init_x(x_in, diffusion_type, vocab_size, batch_size, C, H, W, device)
    batch_size, C, H, W = x.shape
    t_axis = torch.linspace(0, 1, num_steps)

    log_post = torch.full((batch_size, num_classes), math.log(1.0 / num_classes), device=device)

    xs, post_probs_list, unnorm_post_probs_list, raw_logratios_list = [], [], [], []
    for t, t_next in zip(t_axis[:-1], t_axis[1:]):
        kappa_t,      _ = get_scheduler(t,      scheduler_type=scheduler_type, polynomial_power=polynomial_power)
        kappa_t_next, _ = get_scheduler(t_next, scheduler_type=scheduler_type, polynomial_power=polynomial_power)
        kappa_ts = kappa_t * torch.ones(batch_size, device=device)
        xs.append(x.clone())

        # Sampling step
        with torch.no_grad():
            logits_cond   = cond_model(x, kappa_ts, c)
            logits_uncond = uncond_model(x, kappa_ts, c_uncond)
        x_pred = mix_scores(logits_cond, logits_uncond, guid_scale=guid_scale)

        is_masked = get_masked_pixels(x, vocab_size)
        x_next, _ = denoising_step(
            x_pred, x, t, t_next, is_masked, vocab_size, diffusion_type,
            scheduler_type=scheduler_type, polynomial_power=polynomial_power,
            use_ancestral_sampling=use_ancestral_sampling,
        )

        # Score the transition under all K classes (one batched pass) and the unconditional model.
        # The unconditional prediction is the one already computed for CFG (same x, kappa_t, c_uncond).
        labels_k      = torch.arange(num_classes, device=device).repeat_interleave(batch_size)
        x_rep_k       = x.unsqueeze(0).repeat(num_classes, 1, 1, 1, 1).view(num_classes * batch_size, C, H, W)
        x_next_rep_k  = x_next.unsqueeze(0).repeat(num_classes, 1, 1, 1, 1).view(num_classes * batch_size, C, H, W)
        kappa_ts_rep_k = kappa_ts.repeat(num_classes)

        with torch.no_grad():
            x_pred_k  = torch.softmax(cond_model(x_rep_k, kappa_ts_rep_k, labels_k), dim=-1)  # [K*B, C, H, W, V]
        x_pred_uncond = torch.softmax(logits_uncond, dim=-1)                                  # [B, C, H, W, V]

        distr_k      = compute_transition_distr(x_pred_k,      x_rep_k, kappa_t, kappa_t_next, vocab_size, diffusion_type, use_ancestral_sampling_probs)
        distr_uncond = compute_transition_distr(x_pred_uncond, x,       kappa_t, kappa_t_next, vocab_size, diffusion_type, use_ancestral_sampling_probs)

        log_p_classes = gather_token_transition_logprobs(distr_k, x_next_rep_k).sum(dim=[1, 2, 3]).view(num_classes, batch_size)  # [K, B]
        log_p_uncond  = gather_token_transition_logprobs(distr_uncond, x_next).sum(dim=[1, 2, 3])                                 # [B]
        step_logratio = (log_p_classes - log_p_uncond.unsqueeze(0)).T                                                            # [B, K]

        if normalize_by_unmask_count and diffusion_type == 'masking':
            newly_unmasked  = is_masked & ~get_masked_pixels(x_next, vocab_size)
            n_unmasked_step = newly_unmasked.flatten(1).sum(-1).clamp_min(1)  # [B]
            step_logratio   = step_logratio / n_unmasked_step.unsqueeze(-1)

        log_post = log_post + temp * step_logratio + offset
        unnorm_post_probs_list.append(torch.exp(log_post.detach().clone()))
        post_probs_list.append(torch.softmax(log_post, dim=-1).detach().clone())
        raw_logratios_list.append(step_logratio.detach().clone())

        x = x_next

    return (x, torch.stack(xs), torch.stack(post_probs_list),
            torch.stack(unnorm_post_probs_list), torch.stack(raw_logratios_list))
