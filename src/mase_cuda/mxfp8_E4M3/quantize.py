import torch


def quantize1d_E4M3_simulated(weights: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    assert weights.ndim == 1, "Weights tensor must be 1D"
    assert group_size > 0, "Group size must be positive"
    numel = weights.numel()
    assert numel % group_size == 0, "Number of elements in the weights tensor must be divisible by the group size"

    num_groups = numel // group_size
    weights = weights.bfloat16().float()
    w_g = weights.flatten().reshape(num_groups, group_size)

    # Get the sign bit
    sign = torch.where(w_g < 0, torch.tensor(-1, dtype=torch.int8), torch.tensor(1, dtype=torch.int8))
    w_g = w_g.abs()

    # Get the exponent bits
    exponent = (w_g.view(torch.int32) >> 23) & 0xFF
    group_exp = exponent.max(dim=1, keepdim=True).values
    scales = group_exp.to(torch.uint8).flatten()
    
    w_g = w_g.to(torch.bfloat16).view(torch.int16)
    w_g_exp = (w_g & 0x7F80) >> 7
    w_g_exp = group_exp - w_g_exp 
    w_g_exp = torch.where(w_g_exp > 15, 15, w_g_exp)
    w_g_exp = (w_g_exp << 3 ) & 0x78
    w_g_flag = (w_g & 0x8) >> 3
    w_g_frac = (((w_g & 0x70) >> 4 ) + w_g_flag)
    w_g_frac = torch.where(w_g_frac > 0x7, 0x7, w_g_frac)  # avoid overflow
    sign = sign & 0x80
    w_g = (sign|w_g_exp| w_g_frac).to(torch.uint8)

    mantissa = w_g.flatten()
    return mantissa, scales

