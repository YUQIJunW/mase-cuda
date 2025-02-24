import torch


def quantize1d_simulated(weights: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    assert weights.ndim == 1, "Weights tensor must be 1D"
    bias = 0x7
    final_bias = 0x7F - bias
    assert group_size > 0, "Group size must be positive"
    numel = weights.numel()
    assert numel % group_size == 0, "Number of elements in the weights tensor must be divisible by the group size"

    num_groups = numel // group_size
    weights = weights.bfloat16().float()
    print(weights)
    w_g = weights.flatten().reshape(num_groups, group_size)

    sign = torch.where(w_g < 0, torch.tensor(-1, dtype=torch.int8), torch.tensor(1, dtype=torch.int8))
    w_g = w_g.abs()
    # w_g = torch.where(w_g < torch.finfo(torch.bfloat16).smallest_normal, 0.0, w_g)
    # is_zeros = torch.all(w_g == 0.0, dim=1, keepdim=True)

    exponent = ((w_g.view(torch.int32) >> 23) & 0xFF)
    group_exp = exponent.max(dim=1, keepdim=True).values - 0x7F
    # group_exp = torch.where(group_exp < 0, 0, group_exp)  # avoids division by zero
    scales = group_exp.to(torch.int8).flatten()
    
    w_g = w_g.to(torch.bfloat16).view(torch.int16)
    w_g_exp = (w_g & 0x7F80) >> 7
    w_g_exp = ((w_g_exp - final_bias - group_exp) << 3 ) & 0x78
    w_g_frac = ((w_g & 0x70) >> 4 ) & 0x07
    w_g = (w_g_exp| w_g_frac).to(torch.int8)

    mantissa = (w_g* sign).flatten()
    print(mantissa.view(torch.float8_e4m3fn))
    print(scales)
    return mantissa, scales
