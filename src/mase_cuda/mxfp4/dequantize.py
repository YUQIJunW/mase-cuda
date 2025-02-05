import torch
# import mase_cuda_ext


# mxfp4 E2M1 dequantize1d
def dequantize1d_E2M1_simulated(input: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    assert input.ndim == 1, "Input tensor must be 1D"
    assert scale.ndim == 1, "Scale tensor must be 1D"
    input = input.view(torch.uint8)
    scale = scale.view(torch.uint8)
    bias = 0x1
    final_bias = 0x7F - bias
    
    numel = input.numel()
    num_groups = numel // group_size

    fp8 = input.reshape(num_groups, group_size)
    scales = scale.reshape(num_groups, 1)
    sign = (fp8 & 0x8).to(torch.int16) << 12  # get the sign bit
    exp = (fp8 & 0x6).to(torch.int16) >> 6  # get the exponent bits
    frac = (fp8 & 0x1).to(torch.int16) << 1  # get the mantissa bits
    
    scales = scales.to(torch.int16)  # get the scale bits
    result = exp + scales + final_bias
    exp = ((result & 0xFF) | ((result >> 8) * 0xFF)) << 7  # add the scale to the exponent

 
    output = (sign | exp | frac).view(torch.bfloat16)

    return output
