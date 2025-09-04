"""
Python wrappers for Nunchaku's quantized GEMV operations.
"""

import torch

from nunchaku._C import ops

# def awq_gemv_w4a16_cuda(
#     in_feats: torch.Tensor,
#     kernel: torch.Tensor,
#     scaling_factors: torch.Tensor,
#     zeros: torch.Tensor,
#     m: int,
#     n: int,
#     k: int,
#     group_size: int = 64,
# ) -> torch.Tensor:
#     output = ops.gemv_awq(
#         in_feats,
#         kernel,
#         scaling_factors,
#         zeros,
#         m,
#         n,
#         k,
#         group_size,
#     )
#     return output


def awq_gemv_w4a16_cuda(
    in_feats: torch.Tensor,
    kernel: torch.Tensor,
    scaling_factors: torch.Tensor,
    zeros: torch.Tensor,
    m: int,
    n: int,
    k: int,
    group_size: int = 64,
) -> torch.Tensor:
    w_packed = kernel.view(torch.int16)
    w_packed = w_packed.reshape(n // 4, k // 64, 4, 16)
    w_packed = w_packed.permute(0, 2, 1, 3)
    w_packed = w_packed.contiguous().view(-1, 8)

    w0 = w_packed & 0x000F
    w1 = (w_packed >> 4) & 0x000F
    w2 = (w_packed >> 8) & 0x000F
    w3 = (w_packed >> 12) & 0x000F

    w_q_unpacked = torch.stack([w0, w1, w2, w3], dim=1)

    w_q = w_q_unpacked.view(n, k)
    w_q_grouped = w_q.view(n, k // group_size, group_size)
    scales_expanded = scaling_factors.transpose(0, 1).unsqueeze(-1)
    zeros_expanded = zeros.transpose(0, 1).unsqueeze(-1)

    w_dequant = w_q_grouped.to(scaling_factors.dtype) * scales_expanded + zeros_expanded
    w_dequant = w_dequant.reshape(n, k)
    output = torch.matmul(in_feats.to(w_dequant.dtype), w_dequant.T)

    return output
