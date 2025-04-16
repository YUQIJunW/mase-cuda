import torch


def quantize1d_E4M3_simulated(weights: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    assert weights.ndim == 1, "Weights tensor must be 1D"
    bias = 0x7
    final_bias = -bias
    assert group_size > 0, "Group size must be positive"
    numel = weights.numel()
    assert numel % group_size == 0, "Number of elements in the weights tensor must be divisible by the group size"

    num_groups = numel // group_size
    weights = weights.bfloat16().float()
    # print(weights)
    w_g = weights.flatten().reshape(num_groups, group_size)

    sign = torch.where(w_g < 0, torch.tensor(-1, dtype=torch.int8), torch.tensor(1, dtype=torch.int8))
    w_g = w_g.abs()
    # w_g = torch.where(w_g < torch.finfo(torch.bfloat16).smallest_normal, 0.0, w_g)
    # is_zeros = torch.all(w_g == 0.0, dim=1, keepdim=True)

    exponent = (w_g.view(torch.int32) >> 23) & 0xFF
    group_exp = exponent.max(dim=1, keepdim=True).values + final_bias
    # group_exp = torch.where(group_exp < 0, 0, group_exp)  # avoids division by zero
    scales = group_exp.to(torch.uint8).flatten()
    
    w_g = w_g.to(torch.bfloat16).view(torch.int16)
    w_g_exp = (w_g & 0x7F80) >> 7
    w_g_exp = w_g_exp - final_bias - group_exp
    w_g_exp = torch.where(w_g_exp < 0, 0, w_g_exp)
    w_g_exp = (w_g_exp << 3 ) & 0x78
    w_g_frac = ((w_g & 0x70) >> 4 ) & 0x07
    sign = sign & 0x80
    w_g = (sign|w_g_exp| w_g_frac).to(torch.uint8)

    mantissa = w_g.flatten()
    # mantissa = (w_g* sign).flatten()
    # print(mantissa.view(torch.float8_e4m3fn))
    # print(scales)
    return mantissa, scales



def test_quantize1d_E4M3_simulated():
    tensor = torch.tensor([-1.1084e-01,  5.3467e-02,  5.5908e-02, -2.4414e-04,  8.6670e-02,
3.8086e-02, -8.3984e-02,  1.0449e-01, -2.0557e-01,  3.4180e-03,
3.8086e-02,  3.7598e-02, -1.6675e-01,  1.5527e-01,  7.4463e-02,
-2.2925e-01], dtype=torch.float16)
    group_size = 8

    mantissa, scale = quantize1d_E4M3_simulated(tensor, group_size)

    print("Example Input Tensor:", tensor)

    print("Example Output Tensor:", mantissa)
    print("Example Scale Tensor:", scale)


# test_quantize1d_E4M3_simulated()