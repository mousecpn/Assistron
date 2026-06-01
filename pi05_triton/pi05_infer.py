import torch
import triton
import triton.language as tl
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer
import time
from pi0_infer import (
    vision_encoder,
    layer_norm_matmul_n256_1152_2048_bias,
    rms_matmul_n_2048_2560_qkv_rope,
    matmul_n_2048_2048_res,
    matmul_n_16384_2048_res,
    rms_matmul_n_2048_16384_gate,
    matmul_small_bias,
    matmul_small_bias_res,
    matmul_small_bias_silu,
    matmul_small_gate,
    matmul_k8_n_256,
    matmul_abT_scale,
)


def get_scheduler(name, eps=1e-1):
    if name == "const":
        return lambda t: 1 + 0 * t # compatible for both float and tensor
    elif name == "linear_decay":
        return lambda t: 1 - t
    elif name == "cosine_decay":
        return lambda t: (torch.cos(t * torch.pi / 2))
    elif name == "exp_decay":
        return lambda t: (torch.exp(-t) - math.exp(-1)) / (1 - math.exp(-1))
    elif name == "linear_ramp": 
        return lambda t: t
    elif name == "cosine_ramp":
        return lambda t: (torch.cos(t * torch.pi / 2) + 1) / 2
    elif name == "exp_ramp":
        return lambda t: (torch.exp(t) - 1) / (math.exp(1) - 1)
    elif name == "linear_ramp_rt": 
        return lambda t: t + 0.4 # When ramping in rt, zero schedule results in div zero
    elif name == "cosine_ramp_rt":
        return lambda t: (torch.cos(t * torch.pi / 2) + 1) / 2 + 0.4  # When ramping in rt, zero schedule results in div zero
    elif name == "exp_ramp_rt":
        return lambda t: (torch.exp(t) - 1) / (math.exp(1) - 1) + 0.4  # When ramping in rt, zero schedule results in div zero
    elif name == "as_score":
        return lambda t: 1 / (t + eps) - 1
    elif name == "as_var":
        return lambda t: (1 / (t + eps) - 1).square()
    elif name == "pigdm":
        return lambda t: ((1 - t) ** 2 / ((1 - t) ** 2 + t ** 2) + 1e-8)
    elif name == "pigdm_gamma":
        return lambda t: torch.sqrt(t / (t ** 2 + (1 - t) ** 2 + 1e-8)) 
    elif name == "manual":
        return lambda t: ((1 - t - 0.1).exp() - 1).clamp(0) # first, counter the effect of an as_score schedule, then add scheduler
    else:
        raise ValueError(f"Schedule {name} not supported")

def get_schedule(name, t, eps=1e-1):
    return get_scheduler(name, eps)(t)


@triton.jit
def matmul_bias(inp_ptr, weight_ptr, out_ptr, bias_ptr, 
                seq_len: tl.constexpr, features: tl.constexpr, hidden: tl.constexpr,
                BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        
        # 初始化累加器为全 0
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        
        # 加载 Bias 并累加
        acc += tl.load(
            bias_ptr + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask = ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other = 0.0
        ).to(tl.float32)
        
        # 矩阵乘法
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features + (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other = 0.0
            )
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask = ((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other = 0.0
            )
            acc = tl.dot(x, w, acc)
            
        # 存储结果
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden)
        )

@triton.jit
def matmul_small_res_gate(inp_ptr, weight_ptr, out_ptr, res_ptr, gate_ptr, seq_len : tl.constexpr, features : tl.constexpr, hidden : tl.constexpr,
                         BLOCK_SIZE_N : tl.constexpr, BLOCK_SIZE_M : tl.constexpr, BLOCK_SIZE_K : tl.constexpr):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        
        acc = tl.load(
            res_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other = 0.0
        ).to(tl.float32)
        matmul_acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features + (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other = 0.0
            )
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask = ((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other = 0.0
            )
            matmul_acc = tl.dot(x, w, matmul_acc)
        
        gate = tl.load(
            gate_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other = 0.0
        ).to(tl.float32)

        acc += matmul_acc * gate
        
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden)
        )

def matmul_k_32_1024_bias(x, weight, bias, out):
    seq_len = x.shape[0]
    matmul_small_bias[((seq_len + 31) // 32) * (1024 // 32),] (
        x, weight, out, bias,
        seq_len = seq_len,
        features = 32,
        hidden = 1024,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_K = 32
    )

@triton.jit
def adarms_norm_kernel(
    x_ptr,
    style_ptr,
    normed_x_ptr, 
    gate_ptr, 
    seq_len: tl.constexpr, 
    features: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    for i in range(pid, seq_len, psize):
        row_x_offset = i * features
        sum_sq = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for j in range(0, features, BLOCK_SIZE):
            cols = j + tl.arange(0, BLOCK_SIZE)
            mask = cols < features
            x_val = tl.load(x_ptr + row_x_offset + cols, mask=mask, other=0.0).to(tl.float32)
            sum_sq += x_val * x_val
        
        rms_factor = tl.rsqrt(tl.sum(sum_sq) / features + 1e-6)

        for j in range(0, features, BLOCK_SIZE):
            cols = j + tl.arange(0, BLOCK_SIZE)
            mask = cols < features
            x_val = tl.load(x_ptr + row_x_offset + cols, mask=mask, other=0.0).to(tl.float32)
            x_norm = x_val * rms_factor
            s_scale = tl.load(style_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            s_shift = tl.load(style_ptr + features + cols, mask=mask, other=0.0).to(tl.float32)
            s_gate = tl.load(style_ptr + 2 * features + cols, mask=mask, other=0.0).to(tl.float32)

            output_val = x_norm * (1.0 + s_scale) + s_shift

            tl.store(normed_x_ptr + row_x_offset + cols, output_val.to(tl.bfloat16), mask=mask)
            tl.store(gate_ptr + row_x_offset + cols, s_gate.to(tl.bfloat16), mask=mask)

def matmul_1_1024_1024_bias_silu(x, weight, bias, out):
    seq_len = x.shape[0]
    matmul_small_bias_silu[((seq_len + 31) // 32) * (1024 // 32),] (
        x, weight, out, bias,
        seq_len = seq_len,
        features = 1024,
        hidden = 1024,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_K = 64
    )

@triton.jit
def matmul_rope_qkv(
    inp_ptr, seq_len: tl.constexpr, features: tl.constexpr, head_dim: tl.constexpr, num_heads: tl.constexpr,
    weight_qkv_ptr, rope_weights_ptr, q_ptr, k_ptr, v_ptr,
    BLOCK_SIZE_M : tl.constexpr = 64, BLOCK_SIZE_N : tl.constexpr = 32, BLOCK_SIZE_K : tl.constexpr = 64,
):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)

    grid_m = triton.cdiv(seq_len, BLOCK_SIZE_M)
    grid_n = triton.cdiv((num_heads + 2) * head_dim, BLOCK_SIZE_N)

    assert head_dim % BLOCK_SIZE_N == 0, f"head_dim {head_dim} must be divisible by BLOCK_SIZE_N {BLOCK_SIZE_N}"

    while pid < grid_m * grid_n:
        pid_m = pid // grid_n
        pid_n = pid % grid_n
        start_i = pid_m * BLOCK_SIZE_M
        start_j = pid_n * BLOCK_SIZE_N
        offs_i = start_i + tl.arange(0, BLOCK_SIZE_M)[:, None]
        offs_j = start_j + tl.arange(0, BLOCK_SIZE_N)[None, :]

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            offs_k = k + tl.arange(0, BLOCK_SIZE_K)
            x = tl.load(
                inp_ptr + offs_i * features + offs_k[None, :],
                mask = (offs_i < seq_len) & (offs_k[None, :] < features),
                other = 0,
            )
            w = tl.load(
                weight_qkv_ptr + offs_k[:, None] * ((num_heads + 2) * head_dim) + offs_j,
                mask = (offs_k[:, None] < features) & (offs_j < (num_heads + 2) * head_dim),
                other = 0
            )
            accumulator = tl.dot(x, w, accumulator)

        if start_j < (num_heads + 1) * head_dim:
            x0, x1 = tl.split(accumulator.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2))
            x_cossin = tl.load(rope_weights_ptr + offs_i * head_dim + offs_j % head_dim, mask = offs_i < seq_len, other = 0)
            x_cos, x_sin = tl.split(x_cossin.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2))
            x0_ = x0 * x_cos - x1 * x_sin
            x1_ = x1 * x_cos + x0 * x_sin
            accumulator = tl.interleave(x0_, x1_)

        accumulator = accumulator.to(tl.bfloat16)

        if start_j < num_heads * head_dim:
            out_ptr = q_ptr
            out_stride = num_heads * head_dim
        elif start_j < (num_heads + 1) * head_dim:
            out_ptr = k_ptr
            out_stride = head_dim
        else:
            out_ptr = v_ptr
            out_stride = head_dim
        tl.store(
            out_ptr + offs_i * out_stride + offs_j % out_stride,
            accumulator,
            mask = (offs_i < seq_len) & (offs_j < (num_heads + 2) * head_dim)
        )
        pid += psize

def matmul_k_1024_2560_qkv_rope(x_normed, weight_qkv, rope_weight, Q, K, V):
    seq_len = x_normed.shape[0]
    matmul_rope_qkv[(128,)](
        x_normed, seq_len, 1024, 256, 8,
        weight_qkv, rope_weight, Q, K, V,
    )

def adarms_norm_style_proj(x, time_emb, mod_w, mod_b, x_normed, gate, style):
    seq_len = x.shape[0]
    adarms_norm_kernel[(seq_len,)](
        x, 
        style,
        x_normed,
        gate,
        seq_len = seq_len, 
        features = 1024, 
        BLOCK_SIZE = 512
    )
def adarms_norm_style_proj_final(x, time_emb, mod_w, mod_b, x_normed, gate, style):
    seq_len = x.shape[0]

    adarms_norm_kernel[(seq_len,)](
        x, 
        style,
        x_normed,
        gate,
        seq_len = seq_len, 
        features = 1024, 
        BLOCK_SIZE = 512
    )


def adarms_matmul_k_1024_32_bias(
    x,
    time_emb,
    mod_w,
    mod_b,
    x_normed,
    gate,
    style,
    weight,
    bias,
    out,
    res,
):
    adarms_norm_style_proj_final(x, time_emb, mod_w, mod_b, x_normed, gate, style)
    seq_len = x.shape[0]
    matmul_bias[((seq_len + 15) // 16) * (32 // 16),] (
        x_normed,
        weight,
        out,
        bias,
        seq_len = seq_len,
        features = 1024,
        hidden = 32,
        BLOCK_SIZE_N = 16,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_K = 256
    )

def adarms_matmul_k_1024_32_bias_res(
    x,
    time_emb,
    mod_w,
    mod_b,
    x_normed,
    gate,
    style,
    weight,
    bias,
    out,
    res,
):
    adarms_norm_style_proj_final(x, time_emb, mod_w, mod_b, x_normed, gate, style)
    seq_len = x.shape[0]
    matmul_small_bias_res[((seq_len + 15) // 16) * (32 // 16),] (
        x_normed,
        weight,
        out,
        bias,
        res,
        seq_len = seq_len,
        features = 1024,
        hidden = 32,
        BLOCK_SIZE_N = 16,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_K = 256
    )

def matmul_k_2048_1024_gate(x, weight, out, gate):
    seq_len = x.shape[0]
    matmul_small_res_gate[(128,)](
        x,
        weight,
        out,
        out, 
        gate,
        seq_len = seq_len,
        features = 2048,
        hidden = 1024,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_K = 128
    )

def matmul_k_4096_1024_gate(x, weight, out, gate):
    seq_len = x.shape[0]
    matmul_small_res_gate[(((seq_len + 15) // 16) * (1024 // 32),)](
        x,
        weight,
        out,
        out,
        gate,
        seq_len = seq_len,
        features = 4096,
        hidden = 1024,
        BLOCK_SIZE_N = 16,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_K = 256
    )

@triton.jit
def softmax_kernel_masklen(
    inp_ptr,
    queries: tl.constexpr,
    keys: tl.constexpr,
    valid_keys_len_ptr,
    out_ptr,
    BLOCK_SIZE_M: tl.constexpr = 4,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)
    big_neg = -2.3819763e38
    assert BLOCK_SIZE >= keys, f"BLOCK_SIZE must be >= keys, got {BLOCK_SIZE} < {keys}"

    valid_keys_len = tl.load(valid_keys_len_ptr).to(tl.int32)
    valid_keys_len = tl.maximum(0, tl.minimum(valid_keys_len, keys))

    for i in range(pid * BLOCK_SIZE_M, queries, psize * BLOCK_SIZE_M):
        offs_i = i + tl.arange(0, BLOCK_SIZE_M)[:, None]
        offs_j = tl.arange(0, BLOCK_SIZE)[None, :]
        attn_mask = (offs_i < queries) & (offs_j < keys) & (offs_j < valid_keys_len)
        vals = tl.load(inp_ptr + offs_i * keys + offs_j, mask=attn_mask, other=big_neg)
        vals = tl.exp(vals - tl.max(vals, axis=1, keep_dims=True))
        vsum = tl.sum(vals, axis=1, keep_dims=True, dtype=tl.float32)
        vals = vals / vsum
        tl.store(out_ptr + offs_i * keys + offs_j, vals.to(tl.bfloat16),
                 mask=(offs_i < queries) & (offs_j < keys))

def transformer_encoder(weights, buffers, encoder_seq_len):
    layer_norm_matmul_n256_1152_2048_bias(
        buffers['vision_x'],
        weights['vision_final_norm_w'],
        weights['vision_final_norm_b'],
        weights['encoder_multi_modal_projector_w'],
        weights['encoder_multi_modal_projector_b'],
        buffers['encoder_x'],
        buffers['vision_x_norm']
    )
    for i in range(18):
        rms_matmul_n_2048_2560_qkv_rope(
            buffers['encoder_x'],
            weights['encoder_attn_qkv_w'][i],
            buffers['encoder_rope_weights'],
            buffers['encoder_Q'],
            buffers['encoder_K'][i, :encoder_seq_len],
            buffers['encoder_V'][i, :encoder_seq_len],
            buffers['encoder_x_norm']
        )
        if i != 17:
            scale = 1.0 / (256 ** 0.5)
            total_queries = buffers['encoder_Q'].shape[0]
            total_keys = encoder_seq_len
            matmul_abT_scale[(((total_queries + 31) // 32) * ((total_keys + 31) // 32),)](
                buffers['encoder_Q'],
                buffers['encoder_K'][i, :encoder_seq_len],
                buffers['encoder_logits_buf'],
                total_queries,
                total_keys,
                256,
                scale,
                BLOCK_SIZE_M=32,
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=64,
            )
            softmax_kernel_masklen[((total_queries + 3) // 4,)](
                buffers['encoder_logits_buf'],
                total_queries,
                total_keys,
                buffers['valid_encoder_len'],
                buffers['encoder_attn_buf'],
                BLOCK_SIZE_M=4,
                BLOCK_SIZE=1024,
            )
            matmul_k8_n_256(
                buffers['encoder_attn_buf'],
                buffers['encoder_V'][i, :encoder_seq_len],
                buffers['encoder_ctx_buf'],
            )
            
            matmul_n_2048_2048_res(
                buffers['encoder_ctx_buf'].view(-1, 2048),
                weights['encoder_attn_o_w'][i],
                buffers['encoder_x']
            )
        
            rms_matmul_n_2048_16384_gate(
                buffers['encoder_x'],
                weights['encoder_ffn_gate_w'][i],
                weights['encoder_ffn_up_w'][i],
                buffers['encoder_hidden'],
                buffers['encoder_x_norm']
            )

            matmul_n_16384_2048_res(
                buffers['encoder_hidden'],
                weights['encoder_ffn_down_w'][i],
                buffers['encoder_x']
            )

@triton.jit
def softmax_kernel_prefix_suffix(
    inp_ptr,
    queries: tl.constexpr,
    keys_prefix: tl.constexpr,
    keys_suffix: tl.constexpr,
    valid_prefix_len_ptr,
    out_ptr,
    BLOCK_SIZE_M: tl.constexpr = 4,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)
    big_neg = -2.3819763e38
    total_keys: tl.constexpr = keys_prefix + keys_suffix
    assert BLOCK_SIZE >= total_keys, f"BLOCK_SIZE must be >= total_keys, got {BLOCK_SIZE} < {total_keys}"

    valid_prefix_len = tl.load(valid_prefix_len_ptr).to(tl.int32)
    valid_prefix_len = tl.maximum(0, tl.minimum(valid_prefix_len, keys_prefix))

    for i in range(pid * BLOCK_SIZE_M, queries, psize * BLOCK_SIZE_M):
        offs_i = i + tl.arange(0, BLOCK_SIZE_M)[:, None]
        offs_j = tl.arange(0, BLOCK_SIZE)[None, :]

        in_bounds = (offs_i < queries) & (offs_j < total_keys)
        is_prefix = offs_j < keys_prefix
        prefix_ok = is_prefix & (offs_j < valid_prefix_len)
        suffix_ok = (~is_prefix)
        attn_mask = in_bounds & (prefix_ok | suffix_ok)

        vals = tl.load(inp_ptr + offs_i * total_keys + offs_j, mask=attn_mask, other=big_neg)
        vals = tl.exp(vals - tl.max(vals, axis=1, keep_dims=True))
        vsum = tl.sum(vals, axis=1, keep_dims=True, dtype=tl.float32)
        vals = vals / vsum
        tl.store(out_ptr + offs_i * total_keys + offs_j, vals.to(tl.bfloat16), mask=in_bounds)


def transformer_decoder_onestep(weights, buffers, encoder_seq_len, step):
    matmul_k_32_1024_bias(
        buffers['diffusion_noise'],
        weights['decoder_action_in_proj_w'],
        weights['decoder_action_in_proj_b'],
        buffers['decoder_x']
    )
    seq_len = buffers['decoder_x'].shape[0]
    for i in range(18):
        adarms_norm_style_proj(
            buffers['decoder_x'],
            buffers['decoder_time_emb'][step],
            weights['decoder_pre_attn_norm_mod_w'][i],
            weights['decoder_pre_attn_norm_mod_b'][i],
            buffers['x_normed_buf'],
            buffers['gate_buf'],
            buffers['decoder_style_attn'][step, i]
        )
        matmul_k_1024_2560_qkv_rope(
            buffers['x_normed_buf'], 
            weights['decoder_attn_qkv_w'][i],
            buffers['decoder_rope_weights'],
            buffers['decoder_q_buf'],
            buffers['encoder_K'][i, encoder_seq_len:encoder_seq_len + seq_len],
            buffers['encoder_V'][i, encoder_seq_len:encoder_seq_len + seq_len],
        )
        total_queries = buffers['decoder_q_buf'].shape[0]
        prefix_keys = encoder_seq_len
        suffix_keys = seq_len
        total_keys = prefix_keys + suffix_keys

        matmul_abT_scale[(((total_queries + 31) // 32) * ((total_keys + 31) // 32),)](
            buffers['decoder_q_buf'],
            buffers['encoder_K'][i, :encoder_seq_len + seq_len],
            buffers['decoder_logits_buf'],
            total_queries,
            total_keys,
            256,
            256 ** -0.5,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=32,
            BLOCK_SIZE_K=64,
        )

        softmax_kernel_prefix_suffix[((total_queries + 3) // 4,)](
            buffers['decoder_logits_buf'],
            total_queries,
            prefix_keys,
            suffix_keys,
            buffers['valid_encoder_len'],
            buffers['decoder_attn_buf'],
            BLOCK_SIZE_M=4,
            BLOCK_SIZE=1024,
        )

        matmul_k8_n_256(
            buffers['decoder_attn_buf'],
            buffers['encoder_V'][i, :encoder_seq_len + seq_len],
            buffers['decoder_q_buf'],
        )
        matmul_k_2048_1024_gate(
            buffers['decoder_q_buf'].view(-1, 2048),
            weights['decoder_attn_o_w'][i],
            buffers['decoder_x'],
            buffers['gate_buf']
        )
        adarms_norm_style_proj(
            buffers['decoder_x'],
            buffers['decoder_time_emb'][step],
            weights['decoder_pre_ffn_norm_mod_w'][i],
            weights['decoder_pre_ffn_norm_mod_b'][i],
            buffers['x_normed_buf'],
            buffers['gate_buf'],
            buffers['decoder_style_ffn'][step, i]
        )
        seq_len = buffers['decoder_x'].shape[0]
        matmul_small_gate[( (seq_len + 127) // 128, (4096 + 63) // 64 )](
            buffers['x_normed_buf'],
            weights['decoder_ffn_gate_w'][i],
            weights['decoder_ffn_up_w'][i],
            buffers['decoder_hidden'],
            seq_len,
            1024,
            4096,
        )
        matmul_k_4096_1024_gate(
            buffers['decoder_hidden'],
            weights['decoder_ffn_down_w'][i],
            buffers['decoder_x'],
            buffers['gate_buf']
        )

    # adarms_matmul_k_1024_32_bias_res(
    #     buffers['decoder_x'],
    #     buffers['decoder_time_emb'][step],
    #     weights['decoder_final_norm_mod_w'],
    #     weights['decoder_final_norm_mod_b'],
    #     buffers['x_normed_buf'],
    #     buffers['gate_buf'],
    #     buffers['decoder_style_final'][step],
    #     weights['decoder_action_out_proj_w'],
    #     weights['decoder_action_out_proj_b'],
    #     buffers['diffusion_noise'],
    #     buffers['diffusion_noise'],
    # )
    adarms_matmul_k_1024_32_bias(
        buffers['decoder_x'],
        buffers['decoder_time_emb'][step],
        weights['decoder_final_norm_mod_w'],
        weights['decoder_final_norm_mod_b'],
        buffers['x_normed_buf'],
        buffers['gate_buf'],
        buffers['decoder_style_final'][step],
        weights['decoder_action_out_proj_w'],
        weights['decoder_action_out_proj_b'],
        buffers['noise_velocity'],
        buffers['diffusion_noise'],
    )
    
from typing import Literal
import math
PrefixAttentionSchedule = Literal["ones", "zeros", "linear", "exp"]

def unnormalize_actions_torch(actions: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    """Quantile unnormalization: maps [-1, 1] back to [q01, q99]. Operates fully on-device."""
    return (actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01

def normalize_actions_torch(actions: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    """Quantile normalization: maps [q01, q99] to [-1, 1]. Operates fully on-device."""
    return (actions - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

def get_prefix_weights(start: int, end: int, total: int, schedule: PrefixAttentionSchedule) -> torch.Tensor:
    """With start=2, end=6, total=10, the output will be:
    1  1  4/5 3/5 2/5 1/5 0  0  0  0
           ^              ^
         start           end
    `start` (inclusive) is where the chunk starts being allowed to change. `end` (exclusive) is where the chunk stops
    paying attention to the prefix. if start == 0, then the entire chunk is allowed to change. if end == total, then the
    entire prefix is attended to.

    `end` takes precedence over `start` in the sense that, if `end < start`, then `start` is pushed down to `end`. Thus,
    if `end` is 0, then the entire prefix will always be ignored.
    """
    # 因为 start 和 end 都是标量 int，直接用 Python 内置的 min 即可
    start = min(start, end)
    
    # 提前定义好 float32 类型的 arange，防止后续做除法时发生整数除法或类型报错
    t_arange = torch.arange(total, dtype=torch.float32)
    
    if schedule == "ones":
        w = torch.ones(total, dtype=torch.float32)
    elif schedule == "zeros":
        w = (t_arange < start).float()
    elif schedule == "linear" or schedule == "exp":
        # torch.clamp 对应 jnp.clip
        w = torch.clamp((start - 1 - t_arange) / (end - start + 1) + 1, min=0.0, max=1.0)
        if schedule == "exp":
            # math.e 对应 jnp.e, torch.expm1 对应 jnp.expm1
            w = w * torch.expm1(w) / (math.e - 1)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
        
    # torch.where 对应 jnp.where
    return torch.where(t_arange >= end, torch.tensor(0.0), w)
    
def transformer_decoder(weights, buffers, encoder_seq_len, noise_step, d=3, s=8, rtc=False, hajl=False):
    k_sp = 5
    K_ip = 3
    denoise_step = 10
    if hajl is False:
        noise_step = 10
    
    for step in range(0, denoise_step):
        # if hajl:
        # if step <= k_sp and step > K_ip:
        if step == denoise_step - noise_step:
            buffers['diffusion_noise'][:, :8].copy_(buffers['initial_noise'][step, :, :8])
        transformer_decoder_onestep(weights, buffers, encoder_seq_len, step)
        
        ##### separate velocity calculation with noise replacement for better pipelining #####
        
        if rtc:# and step <= K_ip:
            v = buffers['noise_velocity'] * (10)
            tau = buffers['tau_steps'][step]
            buffers['tau'].copy_(tau)
            inv_r2 = ((tau) ** 2 + (tau) ** 2) / (1 - tau) ** 2 
            max_guidance_weight = 5.0
            recovered_A_1 = buffers['diffusion_noise'] + v * (1 - tau)
            weight = buffers['prefix_weight']
            c = torch.nan_to_num((1 - tau) / tau, posinf=max_guidance_weight)
            guidance_weight = torch.clamp(c * inv_r2, max=max_guidance_weight)
            error = buffers['copilot_noise'] - recovered_A_1
            # unnormalized_error = unnormalize_actions_torch(error, buffers['actions_q01'], buffers['actions_q99']) 
            # unnormalized_error[:, :7] = unnormalized_error[:, :7] @ buffers['JKJ'][step]
            # error = normalize_actions_torch(unnormalized_error, buffers['actions_q01'], buffers['actions_q99'])
            v_pigdm = (guidance_weight * error * weight) 
            v_pigdm[:, 8:] *= 0.0
            v_pigdm = (v + v_pigdm)* (1/10) 
            buffers['diffusion_noise'].copy_(buffers['diffusion_noise'] + v_pigdm) ## original implementation
        else:
            buffers['diffusion_noise'].copy_(buffers['noise_velocity'] + buffers['diffusion_noise']) ## original implementation


def pi05_model(weights, buffers, num_views, encoder_seq_len, noise_step, d, s, rtc=False, hajl=False):
    vision_encoder(weights, buffers, num_views)
    transformer_encoder(weights, buffers, encoder_seq_len)
    transformer_decoder(weights, buffers, encoder_seq_len, noise_step, d, s, rtc, hajl)
    


class Pi05Inference:
    def __init__(
        self,
        checkpoint,
        num_views,
        chunk_size,
        noise_step= 10,
        tokenizer_path: str | None = None,
        max_tokenize_len: int = 200,
        discrete_state_input: bool = True,
        max_prompt_text: str | None = None,
        state_dim_for_max_prompt: int | None = None,
        rtc: bool = True,
        hajl: bool = False,
    ):
        self.discrete_state_input = discrete_state_input
        self.tokenizer_path = tokenizer_path
        self.checkpoint = checkpoint
        self.num_views = num_views
        self.chunk_size = chunk_size
        self.rtc = rtc
        self.hajl = hajl
        self.noise_step = noise_step
        self.max_tokenize_len = int(max_tokenize_len)
        if discrete_state_input:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            if max_prompt_text is not None and state_dim_for_max_prompt is not None:
                self.max_prompt_len = self.estimate_max_prompt_len(
                    tokenizer=self.tokenizer,
                    task_prompt=max_prompt_text,
                    state_dim=int(state_dim_for_max_prompt),
                    max_tokenize_len=self.max_tokenize_len,
                    state_token_value=255,
                )
            else:
                self.max_prompt_len = self.max_tokenize_len
        else:
            self.max_prompt_len = len(checkpoint['language_embeds'])
        print(f"max_prompt_len: {self.max_prompt_len}, max_tokenize_len: {self.max_tokenize_len}")
        self.weights = {
            "vision_patch_embedding_w":           torch.empty(14, 14, 3, 1152,        dtype = torch.bfloat16, device = "cuda"),
            "vision_patch_embedding_b":           torch.empty(1152,                   dtype = torch.bfloat16, device = "cuda"),
            "vision_position_embedding":          torch.empty(256, 1152,              dtype = torch.bfloat16, device = "cuda"),
            "vision_attn_qkv_w":                  torch.empty(27, 1152, 3 * 1152,     dtype = torch.bfloat16, device = "cuda"),
            "vision_attn_qkv_b":                  torch.empty(27, 3 * 1152,           dtype = torch.bfloat16, device = "cuda"),
            "vision_attn_o_w":                    torch.empty(27, 1152, 1152,         dtype = torch.bfloat16, device = "cuda"),
            "vision_attn_o_b":                    torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_ffn_up_w":                    torch.empty(27, 1152, 4304,         dtype = torch.bfloat16, device = "cuda"),
            "vision_ffn_up_b":                    torch.empty(27, 4304,               dtype = torch.bfloat16, device = "cuda"),
            "vision_ffn_down_w":                  torch.empty(27, 4304, 1152,         dtype = torch.bfloat16, device = "cuda"),
            "vision_ffn_down_b":                  torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_pre_attn_norm_w":             torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_pre_attn_norm_b":             torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_pre_ffn_norm_w":              torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_pre_ffn_norm_b":              torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_final_norm_w":                torch.empty(1152,                   dtype = torch.bfloat16, device = "cuda"),
            "vision_final_norm_b":                torch.empty(1152,                   dtype = torch.bfloat16, device = "cuda"),

            "encoder_multi_modal_projector_w":    torch.empty(1152, 2048,             dtype = torch.bfloat16, device = "cuda"),
            "encoder_multi_modal_projector_b":    torch.empty(2048,                   dtype = torch.bfloat16, device = "cuda"),
            "encoder_attn_qkv_w":                 torch.empty(18, 2048, 2560,         dtype = torch.bfloat16, device = "cuda"),
            "encoder_attn_o_w":                   torch.empty(18, 2048, 2048,         dtype = torch.bfloat16, device = "cuda"),
            "encoder_ffn_gate_w":                 torch.empty(18, 2048, 16384,        dtype = torch.bfloat16, device = "cuda"),
            "encoder_ffn_up_w":                   torch.empty(18, 2048, 16384,        dtype = torch.bfloat16, device = "cuda"),
            "encoder_ffn_down_w":                 torch.empty(18, 16384, 2048,        dtype = torch.bfloat16, device = "cuda"),

            "decoder_time_embeds":                torch.zeros(10, 1024,                  dtype=torch.bfloat16, device="cuda"),
            "decoder_time_mlp_in_w":              torch.empty(1024, 1024,             dtype = torch.bfloat16, device = "cuda"),
            "decoder_time_mlp_in_b":              torch.empty(1024,                   dtype = torch.bfloat16, device = "cuda"),
            "decoder_time_mlp_out_w":             torch.empty(1024, 1024,             dtype = torch.bfloat16, device = "cuda"),
            "decoder_time_mlp_out_b":             torch.empty(1024,                   dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_in_proj_w":           torch.empty(32, 1024,                      dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_in_proj_b":           torch.empty(1024,                          dtype = torch.bfloat16, device = "cuda"),
            "decoder_pre_attn_norm_mod_w":        torch.empty(18, 1024, 3 * 1024,     dtype = torch.bfloat16, device = "cuda"), 
            "decoder_pre_attn_norm_mod_b":        torch.empty(18, 3 * 1024,           dtype = torch.bfloat16, device = "cuda"),
            "decoder_pre_ffn_norm_mod_w":         torch.empty(18, 1024, 3 * 1024,     dtype = torch.bfloat16, device = "cuda"), 
            "decoder_pre_ffn_norm_mod_b":         torch.empty(18, 3 * 1024,           dtype = torch.bfloat16, device = "cuda"),
            "decoder_attn_qkv_w":                 torch.empty(18, 1024, 2560,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_attn_o_w":                   torch.empty(18, 2048, 1024,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_ffn_gate_w":                 torch.empty(18, 1024, 4096,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_ffn_up_w":                   torch.empty(18, 1024, 4096,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_ffn_down_w":                 torch.empty(18, 4096, 1024,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_out_proj_w":          torch.empty(1024, 32,               dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_out_proj_b":          torch.empty(32,                     dtype = torch.bfloat16, device = "cuda"),
            "decoder_final_norm_mod_w":           torch.empty(1024, 3 * 1024,         dtype=torch.bfloat16, device="cuda"), 
            "decoder_final_norm_mod_b":           torch.empty(3 * 1024,               dtype=torch.bfloat16, device="cuda"), 
            "language_embeds":                    torch.empty(len(checkpoint['language_embeds']), 2048,  dtype = torch.bfloat16, device = "cuda"),

            
        }

        encoder_seq_len = num_views * 256 + self.max_prompt_len
        decoder_seq_len = chunk_size 
        self.rtc = rtc
        self.hajl = hajl
        self.d = 3
        self.s = 8
        

        self.buffers = {
            'observation_images_normalized':      torch.empty(num_views, 224, 224,3,           dtype=torch.bfloat16,   device = "cuda"),
            'diffusion_noise':                    torch.empty(chunk_size, 32,                  dtype = torch.bfloat16, device = "cuda"),
            'noise_velocity':                     torch.empty(chunk_size, 32,                  dtype = torch.bfloat16, device = "cuda"),
            'copilot_noise':                      torch.empty(chunk_size, 32,                  dtype = torch.bfloat16, device = "cuda"),
            'velocity_command':                   torch.empty(chunk_size, 6,                   dtype = torch.bfloat16, device = "cuda"),
            'initial_noise':                      torch.empty(10, chunk_size, 32,                  dtype = torch.bfloat16, device = "cuda"),
            'vision_x':                           torch.empty(num_views, 256, 1152,            dtype = torch.bfloat16, device = "cuda"),
            'vision_x_norm':                      torch.empty(num_views, 256, 1152,            dtype = torch.bfloat16, device = "cuda"),
            'vision_QKV':                         torch.empty(num_views, 256, 3 * 1152,        dtype = torch.bfloat16, device = "cuda"),
            'vision_hidden':                      torch.empty(num_views, 256, 4304,            dtype = torch.bfloat16, device = "cuda"),
            'vision_x_split_k_buf':               torch.empty((num_views * 256 * 1152 * 4,),   dtype = torch.float32, device = "cuda"),
            'encoder_rope_weights':               torch.empty(encoder_seq_len, 256,            dtype = torch.bfloat16, device = "cuda"),
            'encoder_x':                          torch.empty(encoder_seq_len, 2048,           dtype = torch.bfloat16, device = "cuda"),
            'encoder_x_norm':                     torch.empty(encoder_seq_len, 2048,           dtype = torch.bfloat16, device = "cuda"),
            'encoder_K':                          torch.empty(18, encoder_seq_len + decoder_seq_len, 256,   dtype = torch.bfloat16, device = "cuda"),
            'encoder_V':                          torch.empty(18, encoder_seq_len + decoder_seq_len, 256,   dtype = torch.bfloat16, device = "cuda"),
            'encoder_Q':                          torch.empty(encoder_seq_len * 8, 256,        dtype = torch.bfloat16, device = "cuda"),
            'encoder_hidden':                     torch.empty(encoder_seq_len, 16384,          dtype = torch.bfloat16, device = "cuda"),
            'tau':                                torch.empty((1,),                           dtype = torch.bfloat16, device = "cuda"),
            'valid_encoder_len':                  torch.empty((1,),                           dtype = torch.int32, device = "cuda"),
            'encoder_logits_buf':                 torch.empty((encoder_seq_len * 8, encoder_seq_len), dtype=torch.float32,  device="cuda"),
            'encoder_attn_buf':                   torch.empty((encoder_seq_len * 8, encoder_seq_len), dtype=torch.bfloat16, device="cuda"),
            'encoder_ctx_buf':                    torch.empty((encoder_seq_len * 8, 256),     dtype=torch.bfloat16, device="cuda"),
            'decoder_rope_weights':               torch.empty(decoder_seq_len, 256,            dtype = torch.bfloat16, device = "cuda"),
            'decoder_x':                          torch.empty((decoder_seq_len, 1024),         dtype = torch.bfloat16, device = "cuda"),
            'decoder_x_buf':                      torch.empty((decoder_seq_len, 1024),         dtype=torch.bfloat16,  device = "cuda"),
            'decoder_action_buf':                 torch.empty((decoder_seq_len, 32),           dtype = torch.bfloat16, device = "cuda"),
            'decoder_time_emb':                   torch.empty((10, decoder_seq_len, 1024),         dtype = torch.bfloat16, device = "cuda"),
            'decoder_style_attn':                      torch.empty((10, 18, decoder_seq_len, 1024 * 3),         dtype = torch.bfloat16, device = "cuda"), 
            'decoder_style_ffn':                      torch.empty((10, 18, decoder_seq_len, 1024 * 3),         dtype = torch.bfloat16, device = "cuda"), 
            'decoder_style_final':                      torch.empty((10, decoder_seq_len, 1024 * 3),         dtype = torch.bfloat16, device = "cuda"),            
            'decoder_norm_factor_buf':            torch.empty((decoder_seq_len,),              dtype = torch.bfloat16, device = "cuda"),
            'decoder_q_buf':                      torch.empty((decoder_seq_len * 8, 256),      dtype = torch.bfloat16, device = "cuda"),
            'decoder_logits_buf':                 torch.empty((decoder_seq_len * 8, encoder_seq_len + decoder_seq_len), dtype=torch.float32, device="cuda"),
            'decoder_attn_buf':                   torch.empty((decoder_seq_len * 8, encoder_seq_len + decoder_seq_len),  dtype = torch.bfloat16, device = "cuda"),
            'decoder_hidden':                     torch.empty((decoder_seq_len, 4096),         dtype = torch.bfloat16, device = "cuda"),
            'decode_split_k_buf':                 torch.empty((2, decoder_seq_len, 1024),      dtype = torch.float32, device = "cuda"),
            'x_normed_buf':                       torch.empty((decoder_seq_len, 1024),         dtype = torch.bfloat16, device = "cuda"),
            'gate_buf':                           torch.empty((decoder_seq_len, 1024),         dtype = torch.bfloat16, device = "cuda"),
            'actions_q01':                        torch.zeros(32, dtype=torch.bfloat16, device="cuda"),
            'actions_q99':                        torch.ones(32, dtype=torch.bfloat16, device="cuda"),
        }

        # Precompute constants used during CUDA graph capture to avoid allocations and H2D copies.
        self.buffers['tau_steps'] = (torch.arange(10, device="cuda", dtype=torch.float32) / 10.0).to(
            dtype=self.buffers['tau'].dtype
        )

        prefix_weight = get_prefix_weights(self.d, 15 - self.s, 15, "exp").to(device="cuda")
        prefix_weight = prefix_weight[:, None].expand(15, 32).clone()
        prefix_weight[:, 8:] = 0.0
        self.buffers['prefix_weight'] = prefix_weight
        self.recalculate_weight()


        
        prefix_alloc = self.num_views * 256 + self.max_prompt_len
        max_pos = (self.num_views * 256 + self.max_prompt_len - 1) + self.chunk_size
        position_ids = torch.arange(max_pos + 1, device="cuda")
        inv_freq = 1.0 / (10000 ** (torch.arange(0, 256, 2, dtype=torch.float32, device="cuda") / 256))
        k_phase = inv_freq[None, :] * position_ids[:, None]
        k_cos = torch.cos(k_phase).to(torch.bfloat16)
        k_sin = torch.sin(k_phase).to(torch.bfloat16)
        self._rope_table = torch.cat([k_cos[:, :, None], k_sin[:, :, None]], 2).view(-1, 256)
        self.buffers['encoder_rope_weights'].copy_(self._rope_table[:prefix_alloc])

        self.buffers['valid_encoder_len'].fill_(self.num_views * 256 + 1)
        for k, v in checkpoint.items():
            if k != "embedding_weight":
                self.weights[k].copy_(v)
        num_steps = 10
        self.weights['decoder_action_out_proj_w'] *= -1.0 / num_steps
        self.weights['decoder_action_out_proj_b'] *= -1.0 / num_steps 

        for step in range(num_steps):
            matmul_1_1024_1024_bias_silu(
                self.weights['decoder_time_embeds'][step].view(1, -1),
                self.weights['decoder_time_mlp_in_w'],
                self.weights['decoder_time_mlp_in_b'],
                self.buffers['decoder_x_buf']
            )
            matmul_1_1024_1024_bias_silu(
                self.buffers['decoder_x_buf'],
                self.weights['decoder_time_mlp_out_w'],
                self.weights['decoder_time_mlp_out_b'],
                self.buffers['decoder_time_emb'][step]
            )
            for i in range(18):
                matmul_small_bias[((decoder_seq_len + 31) // 32) * (3072 // 32),](
                    self.buffers['decoder_time_emb'][step],
                    self.weights['decoder_pre_attn_norm_mod_w'][i],
                    self.buffers['decoder_style_attn'][step, i],
                    self.weights['decoder_pre_attn_norm_mod_b'][i],
                    seq_len = decoder_seq_len,
                    features = 1024, 
                    hidden = 3072,
                    BLOCK_SIZE_N = 32,
                    BLOCK_SIZE_M = 32,
                    BLOCK_SIZE_K = 32
                )
                matmul_small_bias[((decoder_seq_len + 31) // 32) * (3072 // 32),](
                    self.buffers['decoder_time_emb'][step],
                    self.weights['decoder_pre_ffn_norm_mod_w'][i],
                    self.buffers['decoder_style_ffn'][step, i],
                    self.weights['decoder_pre_ffn_norm_mod_b'][i],
                    seq_len = decoder_seq_len,
                    features = 1024, 
                    hidden = 3072,
                    BLOCK_SIZE_N = 32,
                    BLOCK_SIZE_M = 32,
                    BLOCK_SIZE_K = 32
                )
            matmul_small_bias[((decoder_seq_len + 31) // 32) * (3072 // 32),](
                self.buffers['decoder_time_emb'][step],
                self.weights['decoder_final_norm_mod_w'],
                self.buffers['decoder_style_final'][step],
                self.weights['decoder_final_norm_mod_b'],
                seq_len = decoder_seq_len,
                features = 1024, 
                hidden = 3072,
                BLOCK_SIZE_N = 32,
                BLOCK_SIZE_M = 32,
                BLOCK_SIZE_K = 32
                )
            
        self.prompt_embedding = None
        self._prompt_embed_scale = None
        if self.discrete_state_input:
            if "embedding_weight" not in checkpoint:
                raise KeyError("checkpoint must contain 'embedding_weight' when discrete_state_input=True")
            emb_w = checkpoint["embedding_weight"]
            if isinstance(emb_w, np.ndarray):
                emb_w_t = torch.from_numpy(emb_w)
            else:
                emb_w_t = emb_w
            emb_w_t = emb_w_t.to(device="cuda", dtype=torch.bfloat16, non_blocking=True)
            self.prompt_embedding = nn.Embedding(
                num_embeddings=emb_w_t.shape[0],
                embedding_dim=emb_w_t.shape[1],
                device="cuda",
                dtype=torch.bfloat16,
            )
            with torch.no_grad():
                self.prompt_embedding.weight.copy_(emb_w_t)
            self._prompt_embed_scale = float(emb_w_t.shape[1] ** 0.5)
        self.encoder_seq_len = encoder_seq_len

        self.infer_graph = torch.cuda.CUDAGraph()
        self.record_infer_graph()
        
    def recalculate_weight(self, rtc=False):

        if self.d != 0:
            prefix_weight = get_prefix_weights(self.d, 15-self.s, 15, "exp").to(device="cuda")
            # print("d:", self.d, "s:", self.s, "weight:", prefix_weight)
            prefix_weight = prefix_weight[:, None].expand(15, 32).clone()
            prefix_weight[:, 8:] = 0.0            
            self.buffers['prefix_weight'].copy_(prefix_weight)
        else:
            self.buffers['prefix_weight'] *= 0.0
        if rtc is False:
            self.buffers['prefix_weight'] *= 0.0

         
    
    def measurement_matrix(self,):
        gammas = np.array([0.9**i for i in range(15)], dtype=np.float32) # exp decay
        # linear decay
        # gammas = np.linspace(1.0, 0.1, 15, dtype=np.float32)
        gammas = gammas / np.linalg.norm(gammas)
        H_orig = torch.tensor(gammas).unsqueeze(1).repeat(1, 32)
        H_orig[:, 8:] = 0.0
        H = torch.diag_embed(H_orig).transpose(0, 1).reshape(32, 480)
        HH_T = H @ H.T
        return H, HH_T
        
        
        

    def estimate_max_prompt_len(
        self,
        tokenizer: AutoTokenizer,
        task_prompt: str,
        state_dim: int,
        max_tokenize_len: int = 200,
        state_token_value: int = 255,
    ) -> int:
        task_prompt = task_prompt.strip().replace("_", " ")
        state_str = " ".join([str(int(state_token_value))] * int(state_dim))
        full_prompt = f"Task: {task_prompt}, State: {state_str};\nAction: "
        token_ids = tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=int(max_tokenize_len),
            padding=False,
        )["input_ids"][0]
        return int(token_ids.shape[0])

    def build_prompt_embeds(
        self,
        task_prompt: str,
        state_tokens: np.ndarray
    ) -> tuple[torch.Tensor, int]:
        task_prompt = task_prompt.strip().replace("_", " ")
        state_str = " ".join(map(str, state_tokens.tolist()))
        full_prompt = f"Task: {task_prompt}, State: {state_str};\nAction: "
        token_ids = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokenize_len,
            padding=False,
        )["input_ids"][0].to(device="cuda", non_blocking=True)
        embeds = self.prompt_embedding(token_ids) * self._prompt_embed_scale
        return embeds, int(embeds.shape[0])

    def get_decoder_rope_weights(self, prompt_len: int) -> torch.Tensor:
        start = self.num_views * 256 + prompt_len - 1
        end = start + self.chunk_size
        return self._rope_table[start:end]

    def record_run(self):
        pi05_model(self.weights, self.buffers, self.num_views, self.encoder_seq_len, self.noise_step, self.d, self.s, self.rtc, self.hajl)

    def record_infer_graph(self):
        for _ in range(3):
            self.record_run()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            self.infer_graph.capture_begin()
            self.record_run()
            self.infer_graph.capture_end()

    def forward(
        self,
        observation_images_normalized: torch.Tensor,
        diffusion_noise: torch.Tensor,
        task_prompt: str = None,
        state_tokens: np.ndarray = None,
        **kwargs
    ) -> torch.Tensor:
        if self.discrete_state_input:
            prompt_embeds, prompt_len = self.build_prompt_embeds(
                task_prompt=task_prompt,
                state_tokens=state_tokens
            )
        else:
            prompt_embeds = self.weights['language_embeds']
            prompt_len = self.weights['language_embeds'].shape[0]
        start = self.num_views * 256
        self.buffers['encoder_x'][start : start + prompt_len].copy_(prompt_embeds)
        self.buffers['valid_encoder_len'].fill_(start + prompt_len)
        self.buffers['decoder_rope_weights'].copy_(self.get_decoder_rope_weights(prompt_len))
        self.buffers['observation_images_normalized'].copy_(observation_images_normalized)
        self.buffers['diffusion_noise'].copy_(diffusion_noise)
        for key in kwargs.keys():
            if key in self.buffers:
                if kwargs[key] is not None:
                    self.buffers[key].copy_(kwargs[key])
        self.infer_graph.replay()
        return self.buffers['diffusion_noise']