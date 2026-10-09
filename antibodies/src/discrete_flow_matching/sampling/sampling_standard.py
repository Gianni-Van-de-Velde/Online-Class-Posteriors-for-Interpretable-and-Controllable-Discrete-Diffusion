"""Class-conditional sampling with classifier-free guidance, optionally tracking the class posterior."""
import math
import torch

from ..utils import get_special_token_mask
from ..noise_scheduler import get_scheduler
from .sampling_utils import (
    init_x,
    additive_pad_mask,
    mix_scores,
    denoising_step,
    compute_transition_distr,
    gather_token_transition_logprobs,
)


def sample(model, tokenizer, c, c_uncond,
           guid_scale=1.0,
           x_in=None,
           diffusion_type='uniform',
           scheduler_type='linear',
           polynomial_power=1,
           use_ancestral_sampling=False,
           num_steps=100, vocab_size=27,
           batch_size=4, length=10,
           return_trajectory=True,
           uncond_model=None):
    """
    Class-conditional sampling with logit-space CFG (see mix_scores), at a constant guid_scale:

        guid_scale == 0   ->  unconditional (reduces to logits_uncond; c is ignored)
        guid_scale == 1   ->  plain conditional (reduces to logits_cond; c_uncond is ignored)
        guid_scale  > 1   ->  CFG-amplified conditional

    so a sweep over e.g. [0, 1, 2, 3, 5] covers the unconditional and plain-conditional
    baselines as the two endpoints of one axis.

    Args:
        model        : conditional model, forward(x, t, pad_mask, class_labels) -> MaskedLMOutput.
                       Runs the c-conditioned passes and drives the base generative process.
        uncond_model : runs the c_uncond-conditioned passes; defaults to model (same checkpoint,
                       via its label-dropout uncond token). Set it to a dedicated
                       label_dropout_prob=1.0 checkpoint if one is available.
        c, c_uncond  : [B] class labels, and the uncond token (index == model.config.num_classes)

    Returns:
        final_x : [B, L]
        xs      : [T, B, L] trajectory (None if not return_trajectory)
    """
    if uncond_model is None:
        uncond_model = model
    x_in = init_x(x_in, tokenizer, diffusion_type, batch_size, length, vocab_size, model.device)
    batch_size, length = x_in.shape
    x = x_in
    device = model.device
    attention_mask = additive_pad_mask(x, tokenizer).to(device)

    t_axis = torch.linspace(0, 1, num_steps)
    xs = []

    for t, t_next in zip(t_axis[:-1], t_axis[1:]):
        ts = t * torch.ones(batch_size, device=device)
        x = x.to(device)
        is_not_special, is_masked, special_token_ids = get_special_token_mask(x, tokenizer)

        if return_trajectory:
            xs.append(x.clone())

        with torch.no_grad():
            logits_cond   = model(x, ts, pad_mask=attention_mask, class_labels=c).logits
            logits_uncond = uncond_model(x, ts, pad_mask=attention_mask, class_labels=c_uncond).logits

        x_pred = mix_scores(logits_cond, logits_uncond, guid_scale=guid_scale)

        x = denoising_step(
            x_pred, x, t, t_next, is_not_special, is_masked, special_token_ids,
            tokenizer.mask_token_id, diffusion_type, scheduler_type=scheduler_type,
            polynomial_power=polynomial_power, use_ancestral_sampling=use_ancestral_sampling,
        )

    if not return_trajectory:
        return x, None
    return x, torch.stack(xs)


def sample_with_posterior(cond_model, tokenizer, c_uncond, num_classes,
                          x_in=None,
                          diffusion_type='uniform',
                          scheduler_type='linear',
                          polynomial_power=1,
                          use_ancestral_sampling=False,
                          num_steps=100, vocab_size=27,
                          batch_size=4, length=10,
                          temp=1.0, offset=0.0,
                          return_trajectory=True,
                          use_ancestral_sampling_probs=False,
                          uncond_model=None):
    """
    Unconditional sampling while tracking the class posterior p(k | x_t) for all K classes.

    Generation is purely unconditional: every generative step is driven by uncond_model and the
    uncond token alone. The posterior is a pure read-out layered on top and never steers the
    trajectory (unlike sample_NOT's dynamic guid_scale). Each sampled transition x_t -> x_next
    is scored under every class-conditional model and under the unconditional one, and the
    log-ratios accumulate from the uniform prior log(1/K):

        log_post_k += temp * (log q(x_next | x_t, k) - log q(x_next | x_t, uncond)) + offset

    raw_logratios is returned before temp and offset are applied, so the posterior can be
    recomputed for any temp/offset without re-sampling.

    Costs num_classes + 1 forward passes per step: one batched cond_model call over every class
    (scoring only), plus the uncond_model call that both generates and supplies the baseline.

    Returns:
        final_x           : [B, L]
        xs                : [T, B, L] trajectory (None if not return_trajectory)
        post_probs        : [T, B, K] softmax(log_post) after each step
        unnorm_post_probs : [T, B, K] exp(log_post) after each step
        raw_logratios     : [T, B, K] per-step log-ratios, before temp and offset
    """
    if uncond_model is None:
        uncond_model = cond_model
    x_in = init_x(x_in, tokenizer, diffusion_type, batch_size, length, vocab_size, uncond_model.device)
    batch_size, length = x_in.shape
    x = x_in
    device = uncond_model.device
    attention_mask = additive_pad_mask(x, tokenizer).to(device)

    t_axis = torch.linspace(0, 1, num_steps)
    log_post = torch.full((batch_size, num_classes), math.log(1.0 / num_classes), device=device)

    xs, post_probs_list, unnorm_post_probs_list, raw_logratios_list = [], [], [], []

    for t, t_next in zip(t_axis[:-1], t_axis[1:]):
        kappa_t,      _ = get_scheduler(t,      scheduler_type=scheduler_type, polynomial_power=polynomial_power)
        kappa_t_next, _ = get_scheduler(t_next, scheduler_type=scheduler_type, polynomial_power=polynomial_power)
        ts = t * torch.ones(batch_size, device=device)
        x = x.to(device)
        is_not_special, is_masked, special_token_ids = get_special_token_mask(x, tokenizer)

        if return_trajectory:
            xs.append(x.clone())

        # Generative step: unconditional only
        with torch.no_grad():
            logits_unc = uncond_model(x, ts, pad_mask=attention_mask, class_labels=c_uncond).logits
        x_pred_unc = torch.softmax(logits_unc, dim=-1)

        x_next = denoising_step(
            x_pred_unc, x, t, t_next, is_not_special, is_masked, special_token_ids,
            tokenizer.mask_token_id, diffusion_type, scheduler_type=scheduler_type,
            polynomial_power=polynomial_power, use_ancestral_sampling=use_ancestral_sampling,
        )

        # Score the transition under all K classes in one batched pass
        labels_k   = torch.arange(num_classes, device=device).repeat_interleave(batch_size)
        x_rep_k    = x.unsqueeze(0).repeat(num_classes, 1, 1).view(num_classes * batch_size, length)
        ts_rep_k   = ts.repeat(num_classes)
        attn_rep_k = attention_mask.repeat(num_classes, 1)

        with torch.no_grad():
            logits_k = cond_model(x_rep_k, ts_rep_k, pad_mask=attn_rep_k, class_labels=labels_k).logits
        x_pred_k = torch.softmax(logits_k, dim=-1)  # [K*B, L, V]

        is_not_special_rep = is_not_special.repeat(num_classes, 1)
        is_masked_rep      = is_masked.repeat(num_classes, 1)
        x_next_rep_k = x_next.unsqueeze(0).repeat(num_classes, 1, 1).view(num_classes * batch_size, length)

        distr_k   = compute_transition_distr(x_pred_k,   x_rep_k, kappa_t, kappa_t_next, vocab_size, diffusion_type,
                                             is_not_special=is_not_special_rep, is_masked=is_masked_rep,
                                             use_ancestral_sampling_probs=use_ancestral_sampling_probs)
        distr_unc = compute_transition_distr(x_pred_unc, x,       kappa_t, kappa_t_next, vocab_size, diffusion_type,
                                             is_not_special=is_not_special, is_masked=is_masked,
                                             use_ancestral_sampling_probs=use_ancestral_sampling_probs)

        log_p_classes = gather_token_transition_logprobs(distr_k,   x_next_rep_k).sum(dim=1).view(num_classes, batch_size)  # [K, B]
        log_p_uncond  = gather_token_transition_logprobs(distr_unc, x_next).sum(dim=1)                                      # [B]
        step_logratio = (log_p_classes - log_p_uncond.unsqueeze(0)).T                                                       # [B, K]

        log_post = log_post + temp * step_logratio + offset
        unnorm_post_probs_list.append(torch.exp(log_post.detach().clone()))
        post_probs_list.append(torch.softmax(log_post, dim=-1).detach().clone())
        raw_logratios_list.append(step_logratio.detach().clone())

        x = x_next

    if not return_trajectory:
        return x, None, None, None, None
    return (x, torch.stack(xs), torch.stack(post_probs_list),
            torch.stack(unnorm_post_probs_list), torch.stack(raw_logratios_list))
