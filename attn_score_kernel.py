import torch
import triton
import triton.language as tl
import math
from minference.ops.block_sparse_flash_attention import _triton_block_sparse_attention
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
def block_sparse_attn_score_kernel(
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

# @torch.compile(fullgraph=True) # Fullgraph=True to integrate with Triton
def get_top_k(q, k, v, hidden_dim, top_k, block_size):
    # TODO: Figure out causal masking and batching later
    # batch_size, head_dim, _, _ = q.shape
    # q_pool = q.reshape((batch_size, head_dim, -1, block_size, hidden_dim)).mean(dim=-2)
    # k_pool = k.reshape((batch_size, head_dim, -1, block_size, hidden_dim)).mean(dim=-2)
    q_pool = q.reshape((-1, block_size, hidden_dim)).mean(dim=-2)
    k_pool = k.reshape((-1, block_size, hidden_dim)).mean(dim=-2)
    p_pool = torch.einsum(f'mk, nk -> mn', q_pool, k_pool)
    block_indices = torch.topk(p_pool, top_k, dim=-1).indices.to(torch.int32).sort(dim=-1).values
    # if not(v == None):
    #     seqlens = torch.tensor([k.shape[-2]], dtype=torch.int32, device=q.device)
    #     sm_scale = hidden_dim ** -0.5
    #     q = q.unsqueeze(0).unsqueeze(0)
    #     k = k.unsqueeze(0).unsqueeze(0)
    #     v = v.unsqueeze(0).unsqueeze(0)
    #     block_indices = block_indices.unsqueeze(0).unsqueeze(0)
    #     return _triton_block_sparse_attention(q, k, v, seqlens, block_indices, sm_scale, block_size, block_size)
    # print("Get_top_k shape: ", ret.shape)
    return block_indices


# @torch.compile(fullgraph=True)
def get_top_k_two_layer(q, k, v, hidden_dim, top_k, block_size, return_completed=False):
    batch_size, n_head, q_len, _ = q.shape

    secondary_block_size = 16
    block_top_k_adj = top_k * 2

    q_pool = q.reshape((batch_size, n_head, -1, block_size, hidden_dim)).mean(dim=-2)
    k_pool = k.reshape((batch_size, n_head, -1, block_size, hidden_dim)).mean(dim=-2)
    blocks = get_top_k(q_pool, k_pool, None, hidden_dim, block_top_k_adj, secondary_block_size)

    q_pool_len = q_pool.shape[-2]
    k_pool_len = k_pool.shape[-2]

    output = torch.zeros((q_pool_len, block_top_k_adj * secondary_block_size), device='cuda')
    grid = (triton.cdiv(q_pool_len, secondary_block_size), )
    k_pool = k_pool.transpose(-1, -2)

    block_sparse_attn_score_kernel[grid](
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

    block_indices = torch.topk(output, top_k, dim=-1).indices.to(torch.int32).sort(dim=-1).values
    # if not(v == None):
    #     seqlens = torch.tensor([k.shape[-2]], dtype=torch.int32, device=q.device)
    #     sm_scale = hidden_dim ** -0.5
    #     q = q.unsqueeze(0).unsqueeze(0)
    #     k = k.unsqueeze(0).unsqueeze(0)
    #     v = v.unsqueeze(0).unsqueeze(0)
    #     block_indices = block_indices.unsqueeze(0).unsqueeze(0)
    #     return _triton_block_sparse_attention(q, k, v, seqlens, block_indices, sm_scale, block_size, block_size)
    # print("New implementation output shape: ", ret.shape)
    return block_indices

def time_func_with_random(num_tokens=256000):
    q_len = num_tokens
    k_len = num_tokens
    h_dim = 128
    top_k = 400
    block_size = 16

    q_rnd, k_rnd, v_rnd = torch.rand((q_len, h_dim), device='cuda'), torch.rand((k_len, h_dim), device='cuda'), torch.rand((k_len, h_dim), device='cuda')

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = get_top_k_two_layer(q_rnd, k_rnd, None, h_dim, top_k, block_size)
    end.record()
    torch.cuda.synchronize()

    two_layer_time = start.elapsed_time(end) / 1000

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = get_top_k(q_rnd, k_rnd, None, h_dim, top_k, block_size)
    end.record()
    torch.cuda.synchronize()

    regular_time = start.elapsed_time(end) / 1000

    print(f"Two layer time: {two_layer_time}, Regular time: {regular_time}, Ratio: {two_layer_time/regular_time}")

if __name__ == "__main__":
    for i in range(10):
        time_func_with_random()