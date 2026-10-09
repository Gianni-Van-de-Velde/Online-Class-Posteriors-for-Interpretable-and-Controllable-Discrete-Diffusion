import torch
import torch.optim as optim


def get_optimizer(model: torch.nn.Module, **kwargs) -> torch.optim.Optimizer:
    """Builds the optimizer named by kwargs["_name_"]; the remaining kwargs (lr, betas, ...) are passed through."""
    match kwargs.pop("_name_"):
        case "Adam":
            return optim.Adam(model.parameters(), **kwargs)
        case "AdamW":
            return optim.AdamW(model.parameters(), **kwargs)
        case name:
            raise ValueError(f"Unsupported optimizer: {name}")
