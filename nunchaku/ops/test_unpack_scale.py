import torch


def pack_scale_standalone(scale: torch.Tensor, warp_n: int, num_lanes: int) -> torch.Tensor:
    assert scale.dtype in (torch.float16, torch.bfloat16), "Input tensor must be fp16 or bf16."
    n = scale.shape[0]

    s_pack_size = min(max(warp_n // num_lanes, 2), 8)
    num_s_lanes = min(num_lanes, warp_n // s_pack_size)
    num_s_packs = warp_n // (s_pack_size * num_s_lanes)
    warp_s = num_s_packs * num_s_lanes * s_pack_size
    assert warp_s == warp_n, "warp_n for scales should be equal to warp_n for weights."

    scale = scale.reshape(n // warp_s, num_s_packs, num_s_lanes // 4, s_pack_size // 2, 4, 2, -1)

    scale = scale.permute(0, 6, 1, 2, 4, 3, 5).contiguous()

    return scale.view(-1)


def unpack_scale(packed_scale: torch.Tensor, n: int, group_size: int) -> torch.Tensor:
    assert packed_scale.dtype in (torch.float16, torch.bfloat16), "Input tensor must be fp16 or bf16."

    s_pack_size = 4
    num_s_lanes = 32
    num_s_packs = 1
    warp_s = num_s_packs * num_s_lanes * s_pack_size

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


def main():
    print("Starting verification for pack_scale and unpack_scale...")
    print("\n" + "=" * 60)
    print("TEST CASE: warp_n = 128")
    print("=" * 60)

    warp_n = 128
    num_lanes = 32

    n = 256
    k_groups = 32
    dtype = torch.float16

    original_scale = torch.randn((n, k_groups), dtype=dtype)
    print(f"Original tensor shape: {original_scale.shape}, dtype: {original_scale.dtype}")

    packed_scale = pack_scale_standalone(original_scale.clone(), warp_n=warp_n, num_lanes=num_lanes)
    print(f"Packed tensor shape:   {packed_scale.shape}, dtype: {packed_scale.dtype}")

    unpacked_scale = unpack_scale(packed_scale.clone(), n=n, group_size=-1)
    print(f"Unpacked tensor shape: {unpacked_scale.shape}, dtype: {unpacked_scale.dtype}")

    are_aligned = torch.allclose(original_scale, unpacked_scale)

    if are_aligned:
        print("\nSUCCESS: Tensors are aligned")
    else:
        print("\nFAILURE: The unpacked tensor does not match the original.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
