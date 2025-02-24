import torch

# import mase_cuda_ext


# mxfp6 E2M3 dequantize1d
def dequantize1d_E2M3_simulated(input: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    assert input.ndim == 1, "Input tensor must be 1D"
    assert scale.ndim == 1, "Scale tensor must be 1D"
    input = input.view(torch.uint8)
    scale = scale.view(torch.int8)
    bias = 0x1
    final_bias = 0x7F - bias

    numel = input.numel()
    num_groups = numel // group_size

    fp8 = input.reshape(num_groups, group_size)
    scales = scale.reshape(num_groups, 1)
    sign = (fp8 & 0x10).to(torch.int16) << 10  # get the sign bit
    exp = (fp8 & 0x18).to(torch.int16) >> 3  # get the exponent bits
    frac = (fp8 & 0x07).to(torch.int16) << 4  # get the mantissa bits

    scales = scales.to(torch.int16)  # get the scale bits
    result = exp + scales + final_bias
    exp = ((result & 0xFF) | ((result >> 8) * 0xFF)) << 7  # add the scale to the exponent

    output = (sign | exp | frac).view(torch.bfloat16)

    return output


# mxfp6 E3M2 dequantize1d
def dequantize1d_E3M2_simulated(input: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    assert input.ndim == 1, "Input tensor must be 1D"
    assert scale.ndim == 1, "Scale tensor must be 1D"
    input = input.view(torch.uint8)
    scale = scale.view(torch.int8)
    bias = 0x3
    final_bias = 0x7F - bias

    numel = input.numel()
    num_groups = numel // group_size

    fp8 = input.reshape(num_groups, group_size)
    scales = scale.reshape(num_groups, 1)
    sign = (fp8 & 0x20).to(torch.int16) << 10  # get the sign bit
    exp = (fp8 & 0x1C).to(torch.int16) >> 2  # get the exponent bits
    frac = (fp8 & 0x03).to(torch.int16) << 5  # get the mantissa bits

    scales = scales.to(torch.int16)  # get the scale bits
    result = exp + scales + final_bias
    exp = ((result & 0xFF) | ((result >> 8) * 0xFF)) << 7  # add the scale to the exponent

    output = (sign | exp | frac).view(torch.bfloat16)

    return output
