import torch

def pack_weight_standalone(weight: torch.Tensor, bits: int) -> torch.Tensor:
    assert weight.dtype == torch.int32, f"quantized weight should be torch.int32, but got {weight.dtype}."
    n, k = weight.shape

    num_n_packs = 8
    n_pack_size = 2
    num_n_lanes = 8
    reg_n = 1
    num_k_packs = 1
    k_pack_size = 2
    num_k_lanes = 4
    num_k_unrolls = 1 

    mem_n = num_n_packs * n_pack_size * num_n_lanes * reg_n # 128

    if bits == 4:
        reg_k = 8
        mem_k = num_k_packs * k_pack_size * num_k_lanes * reg_k # 64
    elif bits == 8:
        reg_k = 4
        mem_k = num_k_packs * k_pack_size * num_k_lanes * reg_k # 32
    else:
        raise ValueError(f"Unsupported bits: {bits}")
    
    assert n % mem_n == 0, f"output channel size ({n}) should be divisible by mem_n ({mem_n})."
    assert k % (mem_k * num_k_unrolls) == 0, (
        f"input channel size ({k}) should be divisible by "
        f"mem_k ({mem_k}) * num_k_unrolls ({num_k_unrolls})."
    )

    n_tiles, k_tiles = n // mem_n, k // mem_k
    
    weight = weight.reshape(
        n_tiles, num_n_packs, n_pack_size, num_n_lanes, reg_n,
        k_tiles, num_k_packs, k_pack_size, num_k_lanes, reg_k,
    )
    
    weight = weight.permute(0, 5, 6, 1, 3, 8, 2, 7, 4, 9).contiguous()
    
    if bits == 4:
        weight = weight.bitwise_and_(0xF)
        shift = torch.arange(0, 32, 4, dtype=torch.int32, device=weight.device)
        weight = weight.bitwise_left_shift_(shift)
        weight = weight.sum(dim=-1, dtype=torch.int32)
    elif bits == 8:
        weight = weight.bitwise_and_(0xFF)
        shift = torch.arange(0, 32, 8, dtype=torch.int32, device=weight.device)
        weight = weight.bitwise_left_shift_(shift)
        weight = weight.sum(dim=-1, dtype=torch.int32)
    
    return weight.view(dtype=torch.int8)


def unpack_weight(packed_weight: torch.Tensor, n: int, k: int, bits: int) -> torch.Tensor:
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
        mask = unpacked > 127
        
        unpacked[mask] = unpacked[mask].sub_(128)
    else:
        raise NotImplementedError(f"Weight bits {bits} is not supported.")

    unpacked = unpacked.permute(0, 3, 6, 4, 8, 1, 2, 7, 5, 9).contiguous()

    return unpacked.reshape(n, k)



def main():
    print("Starting verification for pack_weight and unpack_weight...")

    print("\n" + "="*50)
    print("TEST CASE: bits = 4")
    print("="*50)
    bits_4 = 4
    n_4, k_4 = 256, 128 
    
    original_weight_4 = torch.randint(-8, 8, (n_4, k_4), dtype=torch.int32)
    print(f"Original tensor shape: {original_weight_4.shape}, dtype: {original_weight_4.dtype}")

    packed_weight_4 = pack_weight_standalone(original_weight_4.clone(), bits=bits_4)
    print(f"Packed tensor shape:   {packed_weight_4.shape}, dtype: {packed_weight_4.dtype}")

    unpacked_weight_4 = unpack_weight(packed_weight_4.clone(), n=n_4, k=k_4, bits=bits_4)
    print(f"Unpacked tensor shape: {unpacked_weight_4.shape}, dtype: {unpacked_weight_4.dtype}")

    are_equal_4 = torch.equal(original_weight_4, unpacked_weight_4)

    if are_equal_4:
        print("\nSUCCESS: 4-bit tensors are aligned")
    else:
        print("\nFAILURE: 4-bit tensor does not match the original.")


    print("\n" + "="*50)
    print("TEST CASE: bits = 8")
    print("="*50)
    bits_8 = 8
    n_8, k_8 = 128, 256
    
    original_weight_8 = torch.randint(-128, 128, (n_8, k_8), dtype=torch.int32)
    print(f"Original tensor shape: {original_weight_8.shape}, dtype: {original_weight_8.dtype}")

    packed_weight_8 = pack_weight_standalone(original_weight_8.clone(), bits=bits_8)
    print(f"Packed tensor shape:   {packed_weight_8.shape}, dtype: {packed_weight_8.dtype}")

    unpacked_weight_8 = unpack_weight(packed_weight_8.clone(), n=n_8, k=k_8, bits=bits_8)
    print(f"Unpacked tensor shape: {unpacked_weight_8.shape}, dtype: {unpacked_weight_8.dtype}")

    are_equal_8 = torch.equal(original_weight_8, unpacked_weight_8)

    if are_equal_8:
        print("\nSUCCESS: 8-bit tensors are aligned")
    else:
        print("\nFAILURE: 8-bit tensor does not match the original.")
    print("\n" + "="*50)


if __name__ == "__main__":
    main()