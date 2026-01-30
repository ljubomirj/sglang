import torch


def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x.unsqueeze(1)
    if x.ndim != 3:
        raise ValueError(f"Expected tensor with 2 or 3 dims, got {x.ndim}")
    return x


def quantize_kv_int8(
    tensor: torch.Tensor, group_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simple per-(token, head, group) symmetric int8 quantization.

    Args:
        tensor: [T, H, D] or [T, D]
        group_size: size of the last-dim group (must divide D)

    Returns:
        q: int8 tensor with same shape as input
        scale: float16 tensor of shape [T, H, D / group_size]
    """
    x = _ensure_3d(tensor)
    t, h, d = x.shape
    if d % group_size != 0:
        raise ValueError(
            f"Last dim {d} must be divisible by group_size={group_size} for int8 KV cache."
        )

    x_fp = x.float()
    x_reshaped = x_fp.view(t, h, d // group_size, group_size)
    max_abs = x_reshaped.abs().amax(dim=-1)
    scale = max_abs / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    q = torch.round(x_reshaped / scale[..., None]).clamp(-128, 127).to(torch.int8)
    q = q.view(t, h, d)
    return q, scale.to(torch.float16)
