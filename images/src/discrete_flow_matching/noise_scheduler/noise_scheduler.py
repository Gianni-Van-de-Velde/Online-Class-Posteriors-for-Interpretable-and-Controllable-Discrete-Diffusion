import torch

def get_scheduler_images(t, scheduler_type, polynomial_power=None):
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
        raise ValueError(f"scheduler_type='{scheduler_type}' requires polynomial_power (noise.polynomial_power)")

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
        # Slow-start, fast-end: small steps while the image is still very noisy.
        kappa = 1 - torch.cos(torch.pi / 2 * t)
        kappa_dot = torch.pi / 2 * torch.sin(torch.pi / 2 * t)
    elif scheduler_type == 'sin':
        # Fast-start, slow-end.
        kappa = torch.sin(torch.pi / 2 * t)
        kappa_dot = torch.pi / 2 * torch.cos(torch.pi / 2 * t)
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")
    return kappa, kappa_dot

def noising_images(x_clean, t, vocab_size, diffusion_type='uniform', scheduler_type='linear', polynomial_power=None):
    # Create the noisy tensor
    x_noisy = x_clean.clone()

    if diffusion_type.lower() not in ("masking", "uniform"):
        raise ValueError(f"Unknown diffusion_type: {diffusion_type}")

    # Noise level kappa(t): each pixel stays clean with probability kappa(t)
    kappa_t, _ = get_scheduler_images(t, scheduler_type=scheduler_type, polynomial_power=polynomial_power)

    # Compute corruption mask
    corruption_prob = (1 - kappa_t).view(-1, 1,1,1).expand_as(x_clean)
    corrupted_mask = torch.bernoulli(corruption_prob).bool()
    
    if diffusion_type.lower() == "masking":
        # Mask the tokens
        x_noisy[corrupted_mask] = vocab_size - 1
        
    if diffusion_type.lower() == "uniform":
        # Define the valid replacement tokens
        replacement_ids = torch.ones(vocab_size).to(x_clean.device)

        # Replace the corrupted pixels by uniformly random tokens
        random_tokens = torch.multinomial(replacement_ids, torch.numel(x_clean), replacement=True).view(x_clean.size())
        x_noisy[corrupted_mask] = random_tokens[corrupted_mask]
 
    return x_noisy, corrupted_mask
