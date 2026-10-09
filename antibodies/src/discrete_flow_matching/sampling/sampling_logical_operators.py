"""Compositional NOT sampler driven by the online class posterior."""
import math
import torch

from ..utils import get_special_token_mask
from ..noise_scheduler import get_scheduler
from .sampling_utils import (
    init_x,
    additive_pad_mask,
    denoising_step,
    compute_transition_distr,
    gather_token_transition_logprobs,
)


def _prior_weighted_log_post_init(initial_guid_scale, class_priors, c, device):
    """
    Per-sample starting log_post for sample_NOT's use_prior_weighted_init mode.

    Reads initial_guid_scale as the scale a class would get under a BALANCED prior, turns that
    into an over/under-guidance ratio against the true prior odds, and returns
    log(gs_0 / (1 + gs_0)) per sample:

        r       = initial_guid_scale * (K - 1)     # balanced odds are 1/(K-1)
        gs_0(c) = r * p(c) / (1 - p(c))            # from class_priors[c]

    Returns [B], with -inf where gs_0 == 0 (an absent class, or initial_guid_scale == 0),
    matching how the flat path treats initial_guid_scale == 0.
    """
    if class_priors is None:
        raise ValueError("use_prior_weighted_init=True requires class_priors (the [K] class distribution p(c)).")
    priors = torch.as_tensor(class_priors, dtype=torch.float32, device=device).flatten()
    num_classes = priors.numel()
    if num_classes < 2:
        raise ValueError(f"class_priors must cover at least 2 classes, got {num_classes}")
    if (priors < 0).any() or not math.isclose(float(priors.sum()), 1.0, rel_tol=0, abs_tol=1e-3):
        raise ValueError(f"class_priors must be non-negative and sum to 1, got sum={float(priors.sum())}")
    c = c.to(device)
    if int(c.max()) >= num_classes:
        raise ValueError(
            f"class index {int(c.max())} is out of range for class_priors of length {num_classes} "
            "-- class_priors must cover the real classes only (exclude the uncond token)."
        )
    p_c_prior = priors[c]
    if (p_c_prior >= 1.0).any():
        raise ValueError("class_priors assigns probability 1 to a sampled class -- prior odds are infinite.")
    ratio = initial_guid_scale * (num_classes - 1)
    gs_0 = ratio * p_c_prior / (1.0 - p_c_prior)
    p_c_init = gs_0 / (1.0 + gs_0)
    return torch.where(p_c_init > 0, torch.log(p_c_init), torch.full_like(p_c_init, float('-inf')))


def sample_NOT(cond_model, tokenizer, c, c_uncond,
               guid_scale_max=5.0,
               initial_guid_scale=1.5,
               use_prior_weighted_init=False,
               class_priors=None,
               x_in=None,
               diffusion_type='uniform',
               scheduler_type='linear',
               polynomial_power=1,
               use_ancestral_sampling=False,
               num_steps=100, vocab_size=27,
               batch_size=4, length=10,
               temp=1.0,
               offset=0.0,
               return_trajectory=True,
               use_ancestral_sampling_probs=False,
               uncond_model=None):
    """
    Compositional NOT: the unconditional process steered away from class c,

        x_pred_NOT = x_pred_uncond - guid_scale * (x_pred_c - x_pred_uncond),

    clamped to be non-negative and renormalised over the vocabulary. This follows from the
    rate-level NOT gate in the paper, since the denoising step is affine in x_pred.

    guid_scale is set per sample from the unnormalised posterior of c against the
    unconditional model:

        log_post  += temp * (log q(x_next | x_t, c) - log q(x_next | x_t, uncond)) + offset
        p_c        = min(exp(log_post), p_c_max),   p_c_max = guid_scale_max / (1 + guid_scale_max)
        guid_scale = p_c / (1 - p_c)

    log_post starts at log(initial_guid_scale / (1 + initial_guid_scale)), so the first step
    uses initial_guid_scale (0 = no guidance) rather than the maximum.

    offset shifts every step on top of the temp-scaled log-ratio. Because x_next is drawn from
    the unconditional neighbourhood, the per-step increment has a systematically negative
    expectation even for a sample that *is* class c: E[log_p_c - log_p_unc] <= -D_KL(q_unc||q_c)
    by Gibbs' inequality. offset cancels that baseline drift (e.g. offset ~= the estimated
    per-step D_KL), so log_post tracks excess evidence instead. Default 0.0 is the plain form.

    use_prior_weighted_init: log_post is a Bayesian posterior, so it should really start at the
    prior odds p(c)/(1-p(c)). A flat initial_guid_scale therefore only makes sense for balanced
    labels, and V-gene families are heavily imbalanced. With this flag on, initial_guid_scale is
    instead read as the scale a class would get under a balanced prior, which pins down an
    over/under-guidance ratio r relative to the true prior odds (r == 1 is exact Bayes, r > 1
    over-guides); see _prior_weighted_log_post_init. Majority classes then start above
    initial_guid_scale and rare ones below, and a uniform prior reproduces the flat behaviour
    exactly. Only the starting value changes, so this does not interact with temp. gs_0 is still
    clamped by guid_scale_max at the first step, which majority classes can hit immediately for
    large r.

    Args:
        cond_model   : runs the c-conditioned passes
        uncond_model : runs the c_uncond-conditioned passes and drives the base generative
                       process; defaults to cond_model
        c, c_uncond  : [B] label of the class to steer away from, and the uncond token
        class_priors : [K] class distribution p(c) over the real classes (excluding the uncond
                       token), indexed by the entries of c; only read when use_prior_weighted_init

    Returns:
        final_x     : [B, L]
        xs          : [T, B, L] trajectory           (None if not return_trajectory)
        post_probs  : [T, B] clamped p_c used at each step (None if not return_trajectory)
        guid_scales : [T, B] guid_scale used at each step  (None if not return_trajectory)
    """
    if uncond_model is None:
        uncond_model = cond_model
    x_in = init_x(x_in, tokenizer, diffusion_type, batch_size, length, vocab_size, cond_model.device)
    batch_size, length = x_in.shape
    x = x_in
    device = cond_model.device
    attention_mask = additive_pad_mask(x, tokenizer).to(device)

    t_axis = torch.linspace(0, 1, num_steps)

    if initial_guid_scale < 0:
        raise ValueError(f"initial_guid_scale must be >= 0, got {initial_guid_scale}")
    if use_prior_weighted_init:
        log_post = _prior_weighted_log_post_init(initial_guid_scale, class_priors, c, device)
    else:
        p_c_init = initial_guid_scale / (1.0 + initial_guid_scale)
        log_post_init = math.log(p_c_init) if p_c_init > 0 else float('-inf')
        log_post = torch.full((batch_size,), log_post_init, device=device)
    p_c_max = guid_scale_max / (1.0 + guid_scale_max)

    xs, post_probs_list, guid_scales_list = [], [], []

    for t, t_next in zip(t_axis[:-1], t_axis[1:]):
        kappa_t,      _ = get_scheduler(t,      scheduler_type=scheduler_type, polynomial_power=polynomial_power)
        kappa_t_next, _ = get_scheduler(t_next, scheduler_type=scheduler_type, polynomial_power=polynomial_power)
        ts = t * torch.ones(batch_size, device=device)
        x = x.to(device)
        is_not_special, is_masked, special_token_ids = get_special_token_mask(x, tokenizer)

        if return_trajectory:
            xs.append(x.clone())

        p_c = torch.exp(log_post).clamp(max=p_c_max)
        guid_scale = p_c / (1.0 - p_c)

        with torch.no_grad():
            logits_c   = cond_model(x, ts, pad_mask=attention_mask, class_labels=c).logits
            logits_unc = uncond_model(x, ts, pad_mask=attention_mask, class_labels=c_uncond).logits

        x_pred_c   = torch.softmax(logits_c,   dim=-1)
        x_pred_unc = torch.softmax(logits_unc, dim=-1)

        gs = guid_scale.view(batch_size, 1, 1)
        x_pred_not = x_pred_unc - gs * (x_pred_c - x_pred_unc)
        x_pred_not = x_pred_not.clamp(min=0.0)
        x_pred_not = x_pred_not / x_pred_not.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        x_next = denoising_step(
            x_pred_not, x, t, t_next, is_not_special, is_masked, special_token_ids,
            tokenizer.mask_token_id, diffusion_type, scheduler_type=scheduler_type,
            polynomial_power=polynomial_power, use_ancestral_sampling=use_ancestral_sampling,
        )

        distr_c   = compute_transition_distr(x_pred_c,   x, kappa_t, kappa_t_next, vocab_size, diffusion_type,
                                             is_not_special=is_not_special, is_masked=is_masked,
                                             use_ancestral_sampling_probs=use_ancestral_sampling_probs)
        distr_unc = compute_transition_distr(x_pred_unc, x, kappa_t, kappa_t_next, vocab_size, diffusion_type,
                                             is_not_special=is_not_special, is_masked=is_masked,
                                             use_ancestral_sampling_probs=use_ancestral_sampling_probs)

        log_p_c   = gather_token_transition_logprobs(distr_c,   x_next).sum(dim=1)
        log_p_unc = gather_token_transition_logprobs(distr_unc, x_next).sum(dim=1)

        log_post = log_post + temp * (log_p_c - log_p_unc) + offset

        if return_trajectory:
            post_probs_list.append(p_c.detach().clone())
            guid_scales_list.append(guid_scale.detach().clone())

        x = x_next

    if not return_trajectory:
        return x, None, None, None
    return x, torch.stack(xs), torch.stack(post_probs_list), torch.stack(guid_scales_list)
