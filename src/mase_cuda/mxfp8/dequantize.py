import torch
# import mase_cuda_ext


# mxfp8 E4M3 dequantize1d
def dequantize1d_E4M3_simulated(input: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    assert input.ndim == 1, "Input tensor must be 1D"
    assert scale.ndim == 1, "Scale tensor must be 1D"
    input = input.view(torch.uint8)
    scale = scale.view(torch.uint8)
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
    scale = scale.view(torch.uint8)
    bias = 0xF
    final_bias = 0x7F - bias
    
    numel = input.numel()
    num_groups = numel // group_size

    fp8 = input.reshape(num_groups, group_size)
    scales = scale.reshape(num_groups, 1)
    sign = (fp8 & 0x80).to(torch.int16) << 8  # get the sign bit
    exp = (fp8 & 0x7c).to(torch.int16) >> 2  # get the exponent bits
    frac = (fp8 & 0x03).to(torch.int16) << 5  # get the mantissa bits

    scales = scales.to(torch.int16)  # get the scale bits
    result = exp + scales + final_bias
    exp = ((result & 0xFF) | ((result >> 8) * 0xFF)) << 7  # add the scale to the exponent

 
    output = (sign | exp | frac).view(torch.bfloat16)

    return output



def test_dequantize1d_E4M3_simulated():
    input_tensor = torch.tensor([0x1A, 0x2F, 0x3C, 0x4D, 0x5B, 0x6E, 0x44, 0x8A, 0x9D, 0xAF], dtype=torch.uint8).view(torch.float8_e4m3fn)
    scale_tensor = torch.tensor([0x11, 0x12, 0x51, 0x41, 0x33, 0x22, 0x72, 0x48, 0x22, 0x11], dtype=torch.uint8).view(torch.uint8)
    group_size = 1

    output = dequantize1d_E4M3_simulated(input_tensor, scale_tensor, group_size)
    assert output.shape == (10, 1)  # Expected reshaped output shape
    assert output.dtype == torch.bfloat16
    
    expected_output = input_tensor.to(torch.bfloat16)*(2**scale_tensor.to(torch.bfloat16))

    # print("Example Input Tensor:", input_tensor)
    # print("Example Scale Tensor:", scale_tensor)
    # print("Example Output Tensor:", output)
    # print("Example Expected Output Tensor:", expected_output)
    state = 1
    for i in range(len(input_tensor)):
        if(torch.allclose(output[i], expected_output[i], atol=1)==False):
            print("Mismatch at index", i)
            print("Output:", output[i])
            print("Expected Output:", expected_output[i])
            state = 0
    if(state==1):
        print("Test Passed")


test_dequantize1d_E4M3_simulated()