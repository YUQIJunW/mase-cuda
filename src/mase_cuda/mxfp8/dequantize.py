import torch

# import mase_cuda_ext


# mxfp8 E4M3 dequantize1d
def dequantize1d_E4M3_simulated(input: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    assert input.ndim == 1, "Input tensor must be 1D"
    assert scale.ndim == 1, "Scale tensor must be 1D"
    input = input.view(torch.uint8)
    scale = scale.view(torch.int8)
    bias = 0x7
    final_bias = 0x7F - bias
    numel = input.numel()
    num_groups = numel // group_size

    fp8 = input.reshape(num_groups, group_size)
    scales = scale.reshape(num_groups, 1)
    sign = (fp8 & 0x80).to(torch.int16) << 8  # get the sign bit
    exp = (fp8 & 0x78).to(torch.int16) >> 3  # get the exponent bits
    frac = (fp8 & 0x07).to(torch.int16) << 4  # get the mantissa bits

    scales = scales.to(torch.int16)  # get the scale bits
    result = exp + scales + final_bias
    exp = ((result & 0xFF) | ((result >> 8) * 0xFF)) << 7  # add the scale to the exponent

    output = (sign | exp | frac).view(torch.bfloat16)

    return output


# mxfp8 E5M2 dequantize1d
def dequantize1d_E5M2_simulated(input: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    assert input.ndim == 1, "Input tensor must be 1D"
    assert scale.ndim == 1, "Scale tensor must be 1D"
    input = input.view(torch.uint8)
    scale = scale.view(torch.int8)
    bias = 0xF
    final_bias = 0x7F - bias

    numel = input.numel()
    num_groups = numel // group_size

    fp8 = input.reshape(num_groups, group_size)
    scales = scale.reshape(num_groups, 1)
    sign = (fp8 & 0x80).to(torch.int16) << 8  # get the sign bit
    exp = (fp8 & 0x7C).to(torch.int16) >> 2  # get the exponent bits
    frac = (fp8 & 0x03).to(torch.int16) << 5  # get the mantissa bits

    scales = scales.to(torch.int16)  # get the scale bits
    result = exp + scales + final_bias
    exp = result & 0xFF  << 7
    # exp = ((result & 0xFF) | ((result >> 8) * 0xFF)) << 7  # add the scale to the exponent

    output = (sign | exp | frac).view(torch.bfloat16)

    return output
