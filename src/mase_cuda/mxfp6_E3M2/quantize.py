
import torch


def quantize1d_E3M2_simulated(weights: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    assert weights.ndim == 1, "Weights tensor must be 1D"
    bias = 0x7
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
    group_exp = exponent.max(dim=1, keepdim=True).values - bias
    scales = group_exp.to(torch.uint8).flatten()
    
    w_g = w_g.to(torch.bfloat16).view(torch.int16)
    w_g_exp = (w_g & 0x7F80) >> 7
    w_g_exp = w_g_exp - group_exp
    w_g_exp = torch.where(w_g_exp < 0, 0, w_g_exp)
    w_g_exp = (w_g_exp << 2 ) & 0x1C
    w_g_flag = (w_g & 0x10) >> 4
    w_g_frac = (((w_g & 0x60) >> 5 ) + w_g_flag)
    w_g_frac = torch.where(w_g_frac > 0x3, 0x3, w_g_frac)  # avoid overflow
    sign = (sign & 0x80) >> 2
    w_g = (sign|w_g_exp| w_g_frac).to(torch.uint8)

    mantissa = w_g.flatten()
    mantissa = mantissa.view(-1, 4)
    pack1 = mantissa[:, 0] << 2 | (mantissa[:, 3] & 0x30) >> 4
    pack2 = mantissa[:, 1] << 2 | (mantissa[:, 3] & 0xC) >> 2
    pack3 = mantissa[:, 2] << 2 | (mantissa[:, 3] & 0x3) 
    pack = torch.cat((pack1, pack2, pack3), dim=0).flatten()
    return pack, scales

def test_quantize1d_E3M2_simulated():
    weights_tensor = torch.tensor([-0.5, 1.5, -1.0, 2.5, -2.25, 3.0, -3.25, 4.5], dtype=torch.float32)
    group_size = 8

    quantized_weights, scales = quantize1d_E3M2_simulated(weights_tensor, group_size)

    print("Example Weights Tensor:", weights_tensor)
    print("Example Group Size:", group_size)
    print("Quantized Weights:", quantized_weights)
    print("Scales:", scales)

if __name__ == "__main__":
    test_quantize1d_E3M2_simulated()