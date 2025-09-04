import torch


def forward_quant_gemm(
    act: torch.Tensor,
    wgt: torch.Tensor,
    out: torch.Tensor,
    ascales: torch.Tensor,
    wscales: torch.Tensor,
    lora_act_in: torch.Tensor,
    lora_up: torch.Tensor,
    bias: torch.Tensor,
    fp4: bool,
    alpha: float,
    wcscales: torch.Tensor,
    act_unsigned: bool,
) -> None:
    def _dequantize_w4(
        packed_tensor: torch.Tensor, scales: torch.Tensor, is_unsigned: bool, group_size: int = 64
    ) -> torch.Tensor:
        """Dequantizes a 4-bit packed tensor."""
        # The computation dtype is inferred from the scales tensor (e.g., float16)
        comp_dtype = scales.dtype

        # Unpack two 4-bit values from each byte of the int8 tensor
        packed_tensor_int8 = packed_tensor.to(dtype=torch.int8)
        low_nibble = packed_tensor_int8 & 0x0F
        high_nibble = packed_tensor_int8 >> 4

        # Interleave the nibbles to restore the original tensor structure
        dim1, dim2_packed = packed_tensor.shape
        dim2 = dim2_packed * 2
        unpacked_int8 = torch.empty((dim1, dim2), dtype=torch.int8, device=packed_tensor.device)
        unpacked_int8[:, 0::2] = low_nibble
        unpacked_int8[:, 1::2] = high_nibble

        # Convert to float, handling signed vs. unsigned mapping
        # If unsigned, values are in [0, 15].
        # If signed, values are in [-8, 7].
        if is_unsigned:
            unpacked_float = unpacked_int8.to(comp_dtype)
        else:
            # Map values > 7 to the negative range
            unpacked_int8[unpacked_int8 > 7] -= 16
            unpacked_float = unpacked_int8.to(comp_dtype)

        # Apply scales. The scales are stored transposed (K/group, M/N),
        # so we transpose them back and expand.
        scales_t = scales.transpose(0, 1)
        scales_expanded = scales_t.repeat_interleave(group_size, dim=1)

        return unpacked_float * scales_expanded

    # 1. Dequantize activations
    # The quantization scheme (signed/unsigned) for activations depends on the flags.
    # The CUDA kernel asserts !act_unsigned for fp4, so fp4 implies a signed mapping.
    # The python conversion code suggests fp4 maps to unsigned [0,15].
    # We follow the latter as it aligns with the provided fp_quantize behavior.
    is_act_unsigned = act_unsigned or fp4
    act_dequant = _dequantize_w4(act, ascales, is_unsigned=is_act_unsigned)

    # 2. Dequantize weights
    # Weights are assumed to be signed int4, as is common.
    wgt_dequant = _dequantize_w4(wgt, wscales, is_unsigned=False)

    # 3. Perform the main matrix multiplication
    # result = act @ wgt.T
    result = torch.matmul(act_dequant, wgt_dequant.T)

    # 4. Apply per-channel output scales (if provided)
    if wcscales is not None and wcscales.numel() > 0:
        result = result * wcscales.view(1, -1)

    # 5. Add the LoRA branch result
    if lora_act_in is not None and lora_up is not None and lora_act_in.numel() > 0 and lora_up.numel() > 0:
        lora_result = torch.matmul(lora_act_in.to(lora_up.dtype), lora_up.T)
        result = result + lora_result

    # 6. Add bias (if provided)
    if bias is not None and bias.numel() > 0:
        result = result + bias.view(1, -1)

    # 7. Copy the final result into the pre-allocated output tensor
    out.copy_(result)
