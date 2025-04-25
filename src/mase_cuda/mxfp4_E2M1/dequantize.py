import torch
import mase_cuda_ext

def dequantize1d_E2M1(input: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    """Dequantize a 1D input tensor using the given scale tensor and group size.

    :param input: FP4 input mantissa tensor
    :type input: torch.Tensor
    :param scale: uint8 scale tensor
    :type scale: torch.Tensor
    :param group_size: Group size of MXFP4
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
            y_chunk = mase_cuda_ext.mxfp4_E2M1.dequantize1d(x_chunk, scale_chunk, group_size)
            chunks.append(y_chunk)
        output = torch.cat(chunks)
        output = output.reshape(ori_shape)
        input = input.reshape(ori_shape)
    else:
        output = mase_cuda_ext.mxfp4_E2M1.dequantize1d(input, scale, group_size)

    return output


# mxfp4 E2M1 dequantize1d
def dequantize1d_E2M1_simulated(input: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    assert input.ndim == 1, "Input tensor must be 1D"
    assert scale.ndim == 1, "Scale tensor must be 1D"
    input = input.view(torch.uint8)
    scale = scale.view(torch.uint8)
    numel = input.numel()
    num_groups = numel // group_size

    fp4 = input.reshape(num_groups, group_size)
    scales = scale.reshape(num_groups, 1)
    sign = (fp4 & 0x8).to(torch.int16) << 12  # get the sign bit
    exp = (fp4 & 0x6).to(torch.int16) >> 1  # get the exponent bits
    frac = (fp4 & 0x1).to(torch.int16) << 6  # get the mantissa bits

    scales = scales.to(torch.int16)  # get the scale bits
    result = exp + scales
    result = torch.where(result < 0, 0, result)  # avoid negative exponent
    result = torch.where(result > 0xFE, 0xFE, result)  # avoid overflow
    exp = result << 7

    output = (sign | exp | frac).view(torch.bfloat16)
    output = output.flatten()

    return output


