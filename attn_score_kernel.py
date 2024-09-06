import torch
import triton
import triton.language as tl
import math
import pdb

def pytorch_impl(q, k, blocks, block_size=16):
    output = torch.zeros((q.shape[-2], blocks.shape[-1] * block_size), device="cuda")
    for block in range(blocks.shape[0]):
        q_block = q[block * block_size:(block + 1) * block_size, :]
        for i in range(blocks.shape[-1]):
            k_block_idx = blocks[block, i]
            k_block = k[:, k_block_idx * block_size:(k_block_idx + 1) * block_size] # since this is already transposed
            qk = q_block @ k_block
            # print(f"Q shape: {q_block.shape}, K block index: {k_block_idx}, K shape: {k_block.shape}, QK shape: {qk.shape}")
            output[block * block_size:(block + 1) * block_size, i * block_size:(i + 1) * block_size] = qk
    return output

@triton.jit
def block_sparse_attn_kernel(
    q_ptr: tl.tensor, 
    k_ptr: tl.tensor,
    blocks_ptr: tl.tensor,
    o_ptr: tl.tensor,
    stride_q_a, stride_q_b,
    stride_k_a, stride_k_b,
    stride_b_a, stride_b_b,
    stride_o_a, stride_o_b,
    q_len, k_len,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    DIM: tl.constexpr, 
    CHUNK_COUNT: tl.constexpr,
    BLOCK_COUNTS: tl.constexpr,
    seqlen: tl.constexpr # just assume that everything else will be padded
):
    # pdb.set_trace()

    start_m = tl.program_id(0)
    if start_m * BLOCK_SIZE_M >= seqlen:
        return
    
    start_idx_x = start_m * BLOCK_SIZE_M

    query_block_ptr = tl.make_block_ptr(
        base=q_ptr,
        shape=(q_len, DIM),
        strides=(stride_q_a, stride_q_b),
        offsets=(start_idx_x, 0),
        block_shape=(BLOCK_SIZE_M, DIM),
        order=(0, 1)
    )
    query_block = tl.load(query_block_ptr)
    
    for i in range(BLOCK_COUNTS):
        # find what is the block_count in that specific slice
        block_index = tl.load(blocks_ptr + BLOCK_COUNTS * start_m + i)
        block_index = block_index.to(tl.int32)

        k_blocks_ptr = tl.make_block_ptr(
            base=k_ptr,
            shape=(DIM, k_len),
            strides=(stride_k_a, stride_k_b),
            offsets=(0, block_index * BLOCK_SIZE_N),
            block_shape=(DIM, BLOCK_SIZE_N),
            order=(0, 1)
        )
        k_block = tl.load(k_blocks_ptr)

        qk = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
        qk += tl.dot(query_block, k_block)

        o_blocks_ptr = tl.make_block_ptr(
            base=o_ptr,
            shape=(q_len, BLOCK_COUNTS * BLOCK_SIZE_N),
            strides=(stride_o_a, stride_o_b),
            offsets=(start_idx_x, i * BLOCK_SIZE_N),  # Use 'i' for the correct output block placement
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
            order=(0, 1)
        )
        tl.store(o_blocks_ptr, qk)

@torch.compile
def get_top_k(q, k, hidden_dim, top_k, block_size):
    # TODO: Figure out causal masking and batching later
    q_pool = q.reshape((-1, block_size, hidden_dim)).mean(dim=-2)
    k_pool = k.reshape((-1, block_size, hidden_dim)).mean(dim=-2)
    p_pool = torch.einsum(f'mk, nk -> bhmn', q_pool, k_pool)
    ret = torch.topk(p_pool, top_k, dim=-1).indices.to(torch.int32).sort(dim=-1).values
    print("Regular implementation shape: ", ret.shape)
    return ret
    
@torch.compile
def get_top_k_two_layer(q, k, hidden_dim, top_k, block_size):
    secondary_block_size = 16
    q_pool = q.reshape((-1, block_size, hidden_dim)).mean(dim=-2)
    k_pool = k.reshape((-1, block_size, hidden_dim)).mean(dim=-2)
    blocks = get_top_k(q_pool, k_pool, hidden_dim, top_k * 2, secondary_block_size)

    q_pool_len = q_pool.shape[-2]
    k_pool_len = k_pool.shape[-2]

    output = torch.zeros((q_pool_len, 2 * top_k * secondary_block_size), device='cuda')
    grid = (triton.cdiv(q_pool_len, secondary_block_size), )
    k_rnd = k_rnd.transpose(-1, -2)

    block_sparse_attn_kernel[grid](
        q_pool, k_pool, blocks, output, 
        q_pool.stride(0), q_pool.stride(1),
        k_pool.stride(0), k_pool.stride(1),
        blocks.stride(0), blocks.stride(1),
        output.stride(0), output.stride(1),
        q_pool_len, k_pool_len,
        block_size, block_size,
        hidden_dim,
        math.ceil(k_pool_len / secondary_block_size),
        top_k * 2,
        q_pool_len
    )


    ret = torch.topk(output, top_k, dim=-1).indices.to(torch.int32).sort(dim=-1).values
    print("New implementation output shape: ", ret)
    return ret

if __name__ == "__main__":
    q_len = 128
    k_len = 128
    h_dim = 64
    base_top_k = 16
    block_size = 16
    
    q_rnd, k_rnd = torch.rand((q_len, h_dim), device='cuda'), torch.rand((k_len, h_dim), device='cuda')
    blocks = torch.randint(low=0, high=(k_len // block_size), size=(q_len // block_size, base_top_k), device='cuda')
    output = torch.zeros((q_len, base_top_k * block_size), device='cuda')
    grid = (triton.cdiv(q_len, block_size), )

    k_rnd = k_rnd.transpose(-1, -2)
    print(f"Shapes- q_rnd: {q_rnd.shape}, k_rnd: {k_rnd.shape}, blocks: {blocks.shape}, output: {output.shape}, grid: {grid}")

    print("Blocks: ", blocks)

    block_sparse_attn_kernel[grid](q_rnd, k_rnd, blocks, output, 
        q_rnd.stride(0), q_rnd.stride(1),
        k_rnd.stride(0), k_rnd.stride(1),
        blocks.stride(0), blocks.stride(1),
        output.stride(0), output.stride(1),
        q_len, k_len,
        block_size, block_size,
        h_dim,
        math.ceil(k_len / block_size),
        base_top_k,
        q_len
    ) # compare the top-k computation time 

    pytorch_output = pytorch_impl(q_rnd, k_rnd, blocks, block_size=block_size)

    print("Output from triton: ", output)
    print("Output from pytorch: ", pytorch_output)

    print("Are tensors equivalent?: ", torch.allclose(output, pytorch_output, atol=1e-2, rtol=1e-2))