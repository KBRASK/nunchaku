import torch
from ..utils import ceil_divide

def unpack_weight(packed_weight: torch.Tensor, n: int, k: int, bits: int, unsigned = False) -> torch.Tensor:
    assert packed_weight.dtype == torch.int8, f"Packed weight should be torch.int8, but got {packed_weight.dtype}."

    num_n_packs = 8
    n_pack_size = 2
    num_n_lanes = 8
    reg_n = 1
    num_k_packs = 1
    k_pack_size = 2
    num_k_lanes = 4

    mem_n = num_n_packs * n_pack_size * num_n_lanes * reg_n  # 128

    unpacked = packed_weight.contiguous().view(dtype=torch.int32)

    if bits == 4:
        reg_k = 8
        mem_k = num_k_packs * k_pack_size * num_k_lanes * reg_k  # 64
        n_tiles, k_tiles = n // mem_n, k // mem_k

        shape_before_sum = (
            n_tiles,
            k_tiles,
            num_k_packs,
            num_n_packs,
            num_n_lanes,
            num_k_lanes,
            n_pack_size,
            k_pack_size,
            reg_n,
        )
        unpacked = unpacked.reshape(shape_before_sum)

        shift = torch.arange(0, 32, 4, dtype=torch.int32, device=unpacked.device)
        unpacked = unpacked.unsqueeze(-1).bitwise_right_shift(shift).bitwise_and_(0xF)
        if not unsigned:
            mask = unpacked > 7
            unpacked[mask] = unpacked[mask].sub_(16)
    elif bits == 8:
        reg_k = 4
        mem_k = num_k_packs * k_pack_size * num_k_lanes * reg_k  # 32
        n_tiles, k_tiles = n // mem_n, k // mem_k

        shape_before_sum = (
            n_tiles,
            k_tiles,
            num_k_packs,
            num_n_packs,
            num_n_lanes,
            num_k_lanes,
            n_pack_size,
            k_pack_size,
            reg_n,
        )
        unpacked = unpacked.reshape(shape_before_sum)

        shift = torch.arange(0, 32, 8, dtype=torch.int32, device=unpacked.device)
        unpacked = unpacked.unsqueeze(-1).bitwise_right_shift(shift).bitwise_and_(0xFF)
        if not unsigned:
            mask = unpacked > 127
            unpacked[mask] = unpacked[mask].sub_(256)
    else:
        raise NotImplementedError(f"Weight bits {bits} is not supported.")

    unpacked = unpacked.permute(0, 3, 6, 4, 8, 1, 2, 7, 5, 9).contiguous()

    return unpacked.reshape(n, k)



def unpack_lowrank_weight(weight: torch.Tensor, down: bool) -> torch.Tensor:
    """Unpack Low-Rank Weight."""
    c, r = weight.shape
    assert weight.dtype in (torch.float16, torch.bfloat16), f"Unsupported weight dtype {weight.dtype}."
    reg_n, reg_k = 1, 2
    pack_n = 16 * reg_n
    pack_k = 8 * reg_k

    c_packs, r_packs = c // pack_k, r // pack_n

    weight = weight.view(c_packs, r_packs, 8, 4, 2, 2, reg_n, reg_k)
    weight = weight.permute(0, 1, 4, 2, 6, 5, 3, 7).contiguous()
    weight = weight.view(c_packs, r_packs, pack_n, pack_k)

    if down:
        weight = weight.permute(1, 2, 0, 3).contiguous().view(r, c)
    else:
        weight = weight.permute(0, 2, 1, 3).contiguous().view(c, r)
    return weight


def unpack_scale(packed_scale: torch.Tensor, n: int, group_size: int) -> torch.Tensor:
    assert packed_scale.dtype in (torch.float16, torch.bfloat16), "Input tensor must be fp16 or bf16."

    s_pack_size = 4
    num_s_lanes = 32
    num_s_packs = 1
    warp_s = num_s_packs * num_s_lanes * s_pack_size # 128

    if packed_scale.numel() % n != 0:
        raise ValueError(f"Total elements ({packed_scale.numel()}) is not divisible by n ({n}).")
    k_groups = packed_scale.numel() // n

    shape_after_permute = (
        n // warp_s,
        k_groups,
        num_s_packs,
        num_s_lanes // 4,
        4,
        s_pack_size // 2,
        2,
    )
    unpacked_scale = packed_scale.view(shape_after_permute)
    inverse_permute_dims = (0, 2, 3, 5, 4, 6, 1)
    unpacked_scale = unpacked_scale.permute(*inverse_permute_dims).contiguous()
    original_scale = unpacked_scale.view(n, -1)

    return original_scale
def unpack_micro_scale(
    packed_scale: torch.Tensor, 
    n: int, 
    group_size: int
) -> torch.Tensor:
    assert packed_scale.dtype == torch.float8_e4m3fn, "Packed micro scale must be float8_e4m3fn."
    assert group_size == 16, "Currently only support group size 16 for micro-scale unpacking."

    s_pack_size = 4
    num_s_lanes = 4 * 8
    num_s_packs = ceil_divide(128, s_pack_size * num_s_lanes)
    warp_s = num_s_packs * num_s_lanes * s_pack_size
    assert warp_s == 128
    num_k_groups = packed_scale.numel() // n
    shape_after_permute = (
        n // warp_s,
        num_k_groups // (64 // group_size),
        num_s_packs,
        8,
        4,
        s_pack_size,
        64 // group_size
    )

    scale = packed_scale.view(shape_after_permute)
    inverse_permute = (0, 2, 5, 4, 3, 1, 6)
    scale = scale.permute(*inverse_permute).contiguous()

    scale = scale.view(n, -1)
    scale = scale.to(dtype=torch.bfloat16)

    return scale