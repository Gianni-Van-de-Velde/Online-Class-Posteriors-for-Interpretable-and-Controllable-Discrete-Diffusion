import torch


def get_scheduler(t, scheduler_type, polynomial_power=None):
    """
    Returns the noise schedule kappa(t) and its analytical time-derivative kappa_dot(t).

    Args:
        t (Tensor): timestep, either a float value or a tensor of shape [batch_size]
        scheduler_type (str): linear | cosine | sin | polynomial | inv_polynomial
        polynomial_power (float | None): exponent p, required for (inv_)polynomial and ignored otherwise

    Returns:
        kappa, kappa_dot (Tensor): of the same size as t
    """
    if scheduler_type in ('polynomial', 'inv_polynomial') and polynomial_power is None:
        raise ValueError(f"scheduler_type='{scheduler_type}' requires polynomial_power")

    if scheduler_type == 'linear':
        kappa = t
        kappa_dot = 1
    elif scheduler_type == 'polynomial':
        kappa = 1-(1-t)**polynomial_power
        kappa_dot = polynomial_power*(1-t+1e-5)**(polynomial_power-1)
    elif scheduler_type == 'inv_polynomial':
        kappa = t**polynomial_power
        kappa_dot = polynomial_power*(t+1e-5)**(polynomial_power-1)
    elif scheduler_type == 'cosine':
        # MD4's cosine schedule (Shi et al., 2024), in our time convention.
        # Slow-start, fast-end: small steps while the sequence is still very noisy.
        kappa = 1 - torch.cos(torch.pi / 2 * t)
        kappa_dot = torch.pi / 2 * torch.sin(torch.pi / 2 * t)
    elif scheduler_type == 'sin':
        # Fast-start, slow-end.
        kappa = torch.sin(torch.pi / 2 * t)
        kappa_dot = torch.pi / 2 * torch.cos(torch.pi / 2 * t)
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")
    return kappa, kappa_dot


def noising(x_clean, t, tokenizer, diffusion_type='uniform', scheduler_type='linear',
            polynomial_power=None):
    """
    Corrupts a batch of token sequences to noise level kappa(t): each position stays clean with
    probability kappa(t), and is otherwise masked or replaced by a uniformly random amino acid.

    Unlike the image counterpart, <pad>/<bos>/<eos> are never corrupted -- the sequence scaffold
    is held fixed, and only amino-acid positions take part in the diffusion.

    NB: this is pure masking, not BERT-style 80/10/10, and span masking is not implemented.

    Returns x_noisy [B, L] and the boolean corrupted_mask [B, L].
    """
    if diffusion_type.lower() not in ("masking", "uniform"):
        raise ValueError(f"Unknown diffusion_type: {diffusion_type}")

    # Positions that must survive untouched: padding and the sequence delimiters
    pad_mask = x_clean == tokenizer.pad_token_id
    bos_mask = x_clean == tokenizer.bos_token_id
    eos_mask = x_clean == tokenizer.eos_token_id
    special_tokens_mask = pad_mask | bos_mask | eos_mask

    # Create the noisy tensor
    x_noisy = x_clean.clone()

    # Noise level kappa(t): each token stays clean with probability kappa(t)
    kappa_t, _ = get_scheduler(t, scheduler_type=scheduler_type, polynomial_power=polynomial_power)

    # Compute corruption mask
    corruption_prob = (1 - kappa_t).view(-1, 1).expand_as(x_clean)
    corruption_prob = torch.where(special_tokens_mask, 0.0, corruption_prob)
    corrupted_mask = torch.bernoulli(corruption_prob).bool()

    if diffusion_type.lower() == "masking":
        # Mask the tokens
        x_noisy[corrupted_mask] = tokenizer.mask_token_id

    if diffusion_type.lower() == "uniform":
        # Replace by uniformly random amino acids, never by a special token
        replacement_ids = torch.ones((len(tokenizer)))
        for i in tokenizer.special_token_ids:
            replacement_ids[i] = 0
        replacement_ids = replacement_ids.to(x_clean.device)

        random_aa_tokens = torch.multinomial(replacement_ids, torch.numel(x_clean), replacement=True).view(x_clean.size())
        x_noisy[corrupted_mask] = random_aa_tokens[corrupted_mask]

    return x_noisy, corrupted_mask
