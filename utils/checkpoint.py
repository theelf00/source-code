from typing import Any, Dict, Tuple

import torch


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """Extract model state dict from either raw state_dict or rich checkpoint dict."""
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Unsupported checkpoint format")


def load_model_checkpoint(model: torch.nn.Module, path: str, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load checkpoint into model and return metadata if present."""
    checkpoint = torch.load(path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    model.load_state_dict(state_dict)

    metadata: Dict[str, Any] = {}
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}

    return model, metadata
