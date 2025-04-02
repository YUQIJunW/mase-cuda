import torch
import mase_cuda_ext

def dequantize_E4M3_1d(input: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    """Dequantize a 1D input tensor using the given scale tensor and group size.

    :param input: FP8 input mantissa tensor
    :type input: torch.Tensor
    :param scale: uint8 scale tensor
    :type scale: torch.Tensor
    :param group_size: Group size of MXFP8
    :type group_size: int
    :return: Dequantized output tensor, with the same shape as the input tensor
    :rtype: torch.Tensor
    """
    max_num_ctas = 65535  # 65535 is the maximum value for gridDim.x/y/z
    num_ctas_for_chunk = 65408  # 65535 // 128 * 128, assuming blockDim.y = 128
    num_elements = input.numel()
    num_groups = (num_elements + group_size - 1) // group_size
    # grid dim: (group_size // ..., num_groups // 8 ... 128)
    max_grid_dim_y = (num_groups + 7) // 8
    if max_grid_dim_y > max_num_ctas:
        # if the number of groups is too large, we need to split the input tensor into chunks,
        # because the cuda kernel spreads CTAs along both the x and y dimensions
        # and the y dimension may not be enough to cover all the groups
        chunk_size = group_size * 8 * num_ctas_for_chunk
        ori_shape = input.shape
        input = input.flatten()
        num_chunks = (num_elements + chunk_size - 1) // chunk_size
        chunks = []
        for i in range(num_chunks):
            x_chunk = input[i * chunk_size : (i + 1) * chunk_size]
            scale_chunk = scale[i * chunk_size // group_size : (i + 1) * chunk_size // group_size]
            y_chunk = mase_cuda_ext.mxfp8.dequantize1d(x_chunk, scale_chunk, group_size)
            chunks.append(y_chunk)
        output = torch.cat(chunks)
        output = output.reshape(ori_shape)
        input = input.reshape(ori_shape)
    else:
        output = mase_cuda_ext.mxfp8.dequantize1d(input, scale, group_size)

    return output


# mxfp8 E4M3 dequantize1d
def dequantize1d_E4M3_simulated(input: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    assert input.ndim == 1, "Input tensor must be 1D"
    assert scale.ndim == 1, "Scale tensor must be 1D"
    input = input.view(torch.uint8)
    scale = scale.view(torch.uint8)
    bias = 0x7
    final_bias = -bias
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
    final_bias = -bias

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

def test_dequantize1d_E4M3_simulated():
    input_tensor = torch.tensor([61, 189, 188, 188, 189], dtype=torch.uint8).view(torch.float8_e4m3fn)
    scale_tensor = torch.tensor([127, 126, 123, 126, 124], dtype=torch.uint8).view(torch.uint8)
    group_size = 1

    output = dequantize1d_E4M3_simulated(input_tensor, scale_tensor, group_size).to(torch.float32)
    # assert output.shape == (5, 1)  # Expected reshaped output shape
    # assert output.dtype == torch.bfloat16

    expected_output = input_tensor.to(torch.bfloat16) * (2 ** (scale_tensor.to(torch.bfloat16)-127))

    # print("Example Input Tensor:", input_tensor)
    # print("Example Scale Tensor:", scale_tensor)
    print("Example Output Tensor:", output)
    print("Example Expected Output Tensor:", expected_output)
    state = 1
    for i in range(len(input_tensor)):
        if torch.allclose(output[i], expected_output[i], atol=1) == False:
            print("Mismatch at index", i)
            print("Output:", output[i])
            print("Expected Output:", expected_output[i])
            state = 0
    if state == 1:
        print("Test Passed")

# test_dequantize1d_E4M3_simulated()