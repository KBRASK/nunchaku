import torch
import typing as tp


def ceil_divide(a: int, b: int) -> int:
    return (a + b - 1) // b


def pad(
    tensor: tp.Optional[torch.Tensor],
    divisor: int | tp.Sequence[int],
    dim: int | tp.Sequence[int],
    fill_value: float | int = 0,
) -> torch.Tensor:
    if isinstance(divisor, int):
        if divisor <= 1:
            return tensor
    elif all(d <= 1 for d in divisor):
        return tensor
    if tensor is None:
        return None

    shape = list(tensor.shape)
    if isinstance(dim, int):
        assert isinstance(divisor, int)
        shape[dim] = ceil_divide(shape[dim], divisor) * divisor
    else:
        if isinstance(divisor, int):
            divisor = [divisor] * len(dim)
        for d, div in zip(dim, divisor, strict=True):
            shape[d] = ceil_divide(shape[d], div) * div

    result = torch.full(shape, fill_value, dtype=tensor.dtype, device=tensor.device)
    result[[slice(0, extent) for extent in tensor.shape]] = tensor
    return result


def pack_lowrank_weight(weight: torch.Tensor, down: bool) -> torch.Tensor:
    assert weight.dtype in (torch.float16, torch.bfloat16), f"Unsupported weight dtype {weight.dtype}."

    reg_n, reg_k = 1, 2
    n_pack_size, num_n_lanes = 2, 8
    k_pack_size, num_k_lanes = 2, 4

    pack_n = n_pack_size * num_n_lanes * reg_n
    pack_k = k_pack_size * num_k_lanes * reg_k

    if down:
        weight = pad(weight, divisor=(pack_n, pack_k), dim=(0, 1))
        r, c = weight.shape
        r_packs, c_packs = r // pack_n, c // pack_k
        weight = weight.view(r_packs, pack_n, c_packs, pack_k).permute(2, 0, 1, 3)
    else:
        weight = pad(weight, divisor=(pack_n, pack_k), dim=(0, 1))
        c, r = weight.shape
        c_packs, r_packs = c // pack_n, r // pack_k
        weight = weight.view(c_packs, pack_n, r_packs, pack_k).permute(0, 2, 1, 3)

    weight = weight.reshape(c_packs, r_packs, n_pack_size, num_n_lanes, reg_n, k_pack_size, num_k_lanes, reg_k)
    weight = weight.permute(0, 1, 3, 6, 2, 5, 4, 7).contiguous()

    return weight.view(c, r)


def unpack_lowrank_weight(weight: torch.Tensor, down: bool) -> torch.Tensor:
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


def verify_alignment(shape, down, dtype=torch.float16, device="cpu"):
    print("=" * 60)
    print(f"Testing with shape={shape} and down={down}")
    print("-" * 60)

    original_weight = (torch.randn(shape, dtype=dtype, device=device) * 10).round()
    original_shape = original_weight.shape
    print(f"Original tensor shape: {original_shape}")

    packed_weight = pack_lowrank_weight(original_weight.clone(), down=down)
    print(f"Packed tensor shape:   {packed_weight.shape}")

    unpacked_weight = unpack_lowrank_weight(packed_weight.clone(), down=down)
    print(f"Unpacked tensor shape: {unpacked_weight.shape}")

    unpacked_sliced = unpacked_weight[: original_shape[0], : original_shape[1]]

    are_equal = torch.allclose(original_weight, unpacked_sliced, atol=1e-2)

    if are_equal:
        print("\nSUCCESS: Tensors are aligned")
    else:
        print("\nFAILURE: Tensors DO NOT match.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test case 1: proj_down (rank, in_features)
    verify_alignment(shape=(32, 1000), down=True)

    # Test case 2: proj_up (out_features, rank)
    verify_alignment(shape=(4096, 32), down=False)

    # Test case 3: proj_up (out_features, rank)
    verify_alignment(shape=(4090, 30), down=False)
