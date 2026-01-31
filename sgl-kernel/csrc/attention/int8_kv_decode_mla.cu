#include <torch/extension.h>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#else
#include <cuda_runtime.h>
#endif

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <type_traits>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 2;
constexpr int kBlockThreads = kWarpSize * kWarpsPerBlock;
constexpr int kMaxVHeadDim = 1024;
constexpr int kMaxVPerThread = (kMaxVHeadDim + kBlockThreads - 1) / kBlockThreads;
constexpr int kMaxVVecPerThread =
    ((kMaxVHeadDim / 4) + kBlockThreads - 1) / kBlockThreads;
constexpr int kBlockN = 16;
constexpr int kTokensPerWarp = kBlockN / kWarpsPerBlock;

__device__ __forceinline__ float warp_reduce_sum(float val) {
#if defined(__HIPCC__)
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_down(val, offset, kWarpSize);
  }
#else
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
#endif
  return val;
}

__device__ __forceinline__ float warp_broadcast(float val, int src_lane = 0) {
#if defined(__HIPCC__)
  return __shfl(val, src_lane, kWarpSize);
#else
  return __shfl_sync(0xffffffff, val, src_lane);
#endif
}

template <typename T>
__device__ __forceinline__ float to_float(T val) {
  return static_cast<float>(val);
}

template <>
__device__ __forceinline__ float to_float<half>(half val) {
  return __half2float(val);
}

template <typename Q, typename Out>
__global__ void decode_int8_kv_mla_kernel(
    const Q* __restrict__ q,
    const int8_t* __restrict__ k_cache,
    const int8_t* __restrict__ v_cache,
    const half* __restrict__ k_scale,
    const half* __restrict__ v_scale,
    const int32_t* __restrict__ kv_indptr,
    const int32_t* __restrict__ kv_indices,
    Out* __restrict__ out,
    int32_t batch_size,
    int32_t head_num,
    int32_t kv_head_num,
    int32_t qk_head_dim,
    int32_t v_head_dim,
    int32_t kv_group_size,
    int32_t max_tokens,
    int64_t stride_qbs,
    int64_t stride_qh,
    int64_t stride_kbs,
    int64_t stride_kh,
    int64_t stride_vbs,
    int64_t stride_vh,
    int64_t stride_ks,
    int64_t stride_ksh,
    int64_t stride_vs,
    int64_t stride_vsh,
    int64_t stride_obs,
    int64_t stride_oh,
    float sm_scale,
    float logit_cap) {
  int b = blockIdx.x;
  int h = blockIdx.y;
  if (b >= batch_size || h >= head_num) {
    return;
  }

  if (kv_head_num <= 0) {
    return;
  }
  int kv_group_num = head_num / kv_head_num;
  if (kv_group_num <= 0) {
    return;
  }
  int kv_head = h / kv_group_num;
  if (kv_head >= kv_head_num) {
    return;
  }

  int lane = threadIdx.x % kWarpSize;
  int warp_id = threadIdx.x / kWarpSize;

  extern __shared__ char smem[];
  float* q_shared = reinterpret_cast<float*>(smem);
  float* qk_tile = q_shared + qk_head_dim;
  float* p_tile = qk_tile + kBlockN;
  float* ml_shared = p_tile + kBlockN;
  float acc[kMaxVPerThread];
  float acc_vec[kMaxVVecPerThread][4];

  for (int d = threadIdx.x; d < qk_head_dim; d += blockDim.x) {
    q_shared[d] = to_float(q[b * stride_qbs + h * stride_qh + d]);
  }

  __syncthreads();

  if (v_head_dim > kMaxVHeadDim) {
    return;
  }

  bool vec_k = (kv_group_size % 4 == 0) && (qk_head_dim % 4 == 0);
  bool vec_v = (kv_group_size % 4 == 0) && (v_head_dim % 4 == 0);

  int v_per_thread = (v_head_dim + blockDim.x - 1) / blockDim.x;
  for (int i = 0; i < v_per_thread; ++i) {
    acc[i] = 0.0f;
  }
  if (vec_v) {
    int v_vecs_total = v_head_dim / 4;
    int v_vecs_per_thread = (v_vecs_total + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < v_vecs_per_thread; ++i) {
      acc_vec[i][0] = 0.0f;
      acc_vec[i][1] = 0.0f;
      acc_vec[i][2] = 0.0f;
      acc_vec[i][3] = 0.0f;
    }
  }

  int32_t kv_start = kv_indptr[b];
  int32_t kv_end = kv_indptr[b + 1];

  float m = -INFINITY;
  float l = 0.0f;

  for (int32_t kv_idx = kv_start; kv_idx < kv_end; kv_idx += kBlockN) {
    int tile_count = kv_end - kv_idx;
    if (tile_count > kBlockN) {
      tile_count = kBlockN;
    }

    int t_start = warp_id * kTokensPerWarp;
    int t_end = t_start + kTokensPerWarp;
    if (t_end > tile_count) {
      t_end = tile_count;
    }
    for (int t = t_start; t < t_end; ++t) {
      int32_t token_idx = kv_indices[kv_idx + t];
      bool token_valid = token_idx >= 0 && token_idx < max_tokens;

      float qk = -INFINITY;
      if (token_valid) {
        float partial = 0.0f;
        if (vec_k) {
          for (int d = lane * 4; d < qk_head_dim; d += kWarpSize * 4) {
            int scale_idx = d / kv_group_size;
            float scale = __half2float(
                k_scale[token_idx * stride_ks + kv_head * stride_ksh + scale_idx]);
            const int8_t* k_ptr =
                k_cache + token_idx * stride_kbs + kv_head * stride_kh + d;
            int32_t packed = *reinterpret_cast<const int32_t*>(k_ptr);
            int8_t k0 = static_cast<int8_t>(packed & 0xff);
            int8_t k1 = static_cast<int8_t>((packed >> 8) & 0xff);
            int8_t k2 = static_cast<int8_t>((packed >> 16) & 0xff);
            int8_t k3 = static_cast<int8_t>((packed >> 24) & 0xff);
            partial += q_shared[d] * (static_cast<float>(k0) * scale);
            partial += q_shared[d + 1] * (static_cast<float>(k1) * scale);
            partial += q_shared[d + 2] * (static_cast<float>(k2) * scale);
            partial += q_shared[d + 3] * (static_cast<float>(k3) * scale);
          }
        } else {
          for (int d = lane; d < qk_head_dim; d += kWarpSize) {
            int scale_idx = d / kv_group_size;
            float scale = __half2float(
                k_scale[token_idx * stride_ks + kv_head * stride_ksh + scale_idx]);
            int8_t k_val =
                k_cache[token_idx * stride_kbs + kv_head * stride_kh + d];
            partial += q_shared[d] * (static_cast<float>(k_val) * scale);
          }
        }
        qk = warp_reduce_sum(partial);
        qk = warp_broadcast(qk, 0) * sm_scale;
        if (logit_cap > 0.0f) {
          qk = logit_cap * tanhf(qk / logit_cap);
        }
      }
      if (lane == 0) {
        qk_tile[t] = qk;
      }
    }

    __syncthreads();

    float m_tile = -INFINITY;
    float l_tile = 0.0f;
    if (threadIdx.x == 0) {
      for (int t = 0; t < tile_count; ++t) {
        float qk = qk_tile[t];
        if (qk > m_tile) {
          m_tile = qk;
        }
      }
      float sum = 0.0f;
      for (int t = 0; t < tile_count; ++t) {
        float qk = qk_tile[t];
        float p = (qk == -INFINITY) ? 0.0f : expf(qk - m_tile);
        p_tile[t] = p;
        sum += p;
      }
      l_tile = sum;
      ml_shared[0] = m_tile;
      ml_shared[1] = l_tile;
    }

    __syncthreads();

    m_tile = ml_shared[0];
    l_tile = ml_shared[1];

    float m_new = fmaxf(m, m_tile);
    float scale_old = (m == -INFINITY) ? 0.0f : expf(m - m_new);
    float scale_tile = (l_tile > 0.0f) ? expf(m_tile - m_new) : 0.0f;

    for (int i = 0; i < v_per_thread; ++i) {
      acc[i] *= scale_old;
    }
    if (vec_v) {
      int v_vecs_total = v_head_dim / 4;
      int v_vecs_per_thread = (v_vecs_total + blockDim.x - 1) / blockDim.x;
      for (int i = 0; i < v_vecs_per_thread; ++i) {
        acc_vec[i][0] *= scale_old;
        acc_vec[i][1] *= scale_old;
        acc_vec[i][2] *= scale_old;
        acc_vec[i][3] *= scale_old;
      }
    }

    if (scale_tile != 0.0f) {
      for (int t = 0; t < tile_count; ++t) {
        float p = p_tile[t] * scale_tile;
        if (p == 0.0f) {
          continue;
        }
        int32_t token_idx = kv_indices[kv_idx + t];
        bool token_valid = token_idx >= 0 && token_idx < max_tokens;
        if (!token_valid) {
          continue;
        }
        if (vec_v) {
          int v_vecs_total = v_head_dim / 4;
          int acc_idx = 0;
          for (int vec_idx = threadIdx.x; vec_idx < v_vecs_total;
               vec_idx += blockDim.x, ++acc_idx) {
            int d = vec_idx * 4;
            int scale_idx = d / kv_group_size;
            float scale = __half2float(
                v_scale[token_idx * stride_vs + kv_head * stride_vsh + scale_idx]);
            const int8_t* v_ptr =
                v_cache + token_idx * stride_vbs + kv_head * stride_vh + d;
            int32_t packed = *reinterpret_cast<const int32_t*>(v_ptr);
            int8_t v0 = static_cast<int8_t>(packed & 0xff);
            int8_t v1 = static_cast<int8_t>((packed >> 8) & 0xff);
            int8_t v2 = static_cast<int8_t>((packed >> 16) & 0xff);
            int8_t v3 = static_cast<int8_t>((packed >> 24) & 0xff);
            float p_scale = p * scale;
            acc_vec[acc_idx][0] += p_scale * static_cast<float>(v0);
            acc_vec[acc_idx][1] += p_scale * static_cast<float>(v1);
            acc_vec[acc_idx][2] += p_scale * static_cast<float>(v2);
            acc_vec[acc_idx][3] += p_scale * static_cast<float>(v3);
          }
        } else {
          int acc_idx = 0;
          for (int d = threadIdx.x; d < v_head_dim; d += blockDim.x, ++acc_idx) {
            int scale_idx = d / kv_group_size;
            float scale = __half2float(
                v_scale[token_idx * stride_vs + kv_head * stride_vsh + scale_idx]);
            int8_t v_val =
                v_cache[token_idx * stride_vbs + kv_head * stride_vh + d];
            acc[acc_idx] += p * (static_cast<float>(v_val) * scale);
          }
        }
      }
    }

    m = m_new;
    l = l * scale_old + l_tile * scale_tile;
  }

  float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
  if (vec_v) {
    int v_vecs_total = v_head_dim / 4;
    int acc_idx = 0;
    for (int vec_idx = threadIdx.x; vec_idx < v_vecs_total;
         vec_idx += blockDim.x, ++acc_idx) {
      int d = vec_idx * 4;
      float out0 = acc_vec[acc_idx][0] * inv_l;
      float out1 = acc_vec[acc_idx][1] * inv_l;
      float out2 = acc_vec[acc_idx][2] * inv_l;
      float out3 = acc_vec[acc_idx][3] * inv_l;
      if constexpr (std::is_same<Out, half>::value) {
        out[b * stride_obs + h * stride_oh + d] = __float2half(out0);
        out[b * stride_obs + h * stride_oh + d + 1] = __float2half(out1);
        out[b * stride_obs + h * stride_oh + d + 2] = __float2half(out2);
        out[b * stride_obs + h * stride_oh + d + 3] = __float2half(out3);
      } else {
        out[b * stride_obs + h * stride_oh + d] = out0;
        out[b * stride_obs + h * stride_oh + d + 1] = out1;
        out[b * stride_obs + h * stride_oh + d + 2] = out2;
        out[b * stride_obs + h * stride_oh + d + 3] = out3;
      }
    }
  } else {
    int acc_idx = 0;
    for (int d = threadIdx.x; d < v_head_dim; d += blockDim.x, ++acc_idx) {
      float out_val = acc[acc_idx] * inv_l;
      if constexpr (std::is_same<Out, half>::value) {
        out[b * stride_obs + h * stride_oh + d] = __float2half(out_val);
      } else {
        out[b * stride_obs + h * stride_oh + d] = out_val;
      }
    }
  }
}

}  // namespace

void decode_attention_int8_kv_mla(
    at::Tensor q,
    at::Tensor k_cache,
    at::Tensor v_cache,
    at::Tensor k_scale,
    at::Tensor v_scale,
    at::Tensor kv_indptr,
    at::Tensor kv_indices,
    at::Tensor output,
    double sm_scale,
    double logit_cap,
    int64_t kv_group_size) {
  const auto batch_size = q.size(0);
  const auto head_num = q.size(1);
  const auto qk_head_dim = q.size(2);
  const auto v_head_dim = output.size(2);
  const auto kv_head_num = k_cache.size(1);
  const auto max_tokens = k_cache.size(0);

  TORCH_CHECK(q.is_cuda(), "q must be CUDA/HIP");
  TORCH_CHECK(k_cache.is_cuda(), "k_cache must be CUDA/HIP");
  TORCH_CHECK(v_cache.is_cuda(), "v_cache must be CUDA/HIP");
  TORCH_CHECK(k_scale.is_cuda(), "k_scale must be CUDA/HIP");
  TORCH_CHECK(v_scale.is_cuda(), "v_scale must be CUDA/HIP");
  TORCH_CHECK(output.is_cuda(), "output must be CUDA/HIP");
  TORCH_CHECK(kv_indptr.is_cuda(), "kv_indptr must be CUDA/HIP");
  TORCH_CHECK(kv_indices.is_cuda(), "kv_indices must be CUDA/HIP");
  TORCH_CHECK(
      q.scalar_type() == at::kHalf || q.scalar_type() == at::kFloat,
      "q must be float16 or float32");
  TORCH_CHECK(k_cache.scalar_type() == at::kChar, "k_cache must be int8");
  TORCH_CHECK(v_cache.scalar_type() == at::kChar, "v_cache must be int8");
  TORCH_CHECK(k_scale.scalar_type() == at::kHalf, "k_scale must be float16");
  TORCH_CHECK(v_scale.scalar_type() == at::kHalf, "v_scale must be float16");
  TORCH_CHECK(
      output.scalar_type() == at::kHalf || output.scalar_type() == at::kFloat,
      "output must be float16 or float32");
  TORCH_CHECK(kv_indptr.scalar_type() == at::kInt, "kv_indptr must be int32");
  TORCH_CHECK(kv_indices.scalar_type() == at::kInt, "kv_indices must be int32");
  TORCH_CHECK(
      qk_head_dim % kv_group_size == 0,
      "qk_head_dim must be divisible by kv_group_size");
  TORCH_CHECK(
      v_head_dim % kv_group_size == 0,
      "v_head_dim must be divisible by kv_group_size");
  TORCH_CHECK(
      v_head_dim <= kMaxVHeadDim,
      "v_head_dim exceeds kernel limit");

  const dim3 grid(batch_size, head_num, 1);
  const dim3 block(kBlockThreads, 1, 1);
  size_t shared_bytes =
      sizeof(float) * (qk_head_dim + 2 * kBlockN + 2);

  const auto stream = at::cuda::getDefaultCUDAStream();

  const bool q_is_fp16 = q.scalar_type() == at::kHalf;
  const bool out_is_fp16 = output.scalar_type() == at::kHalf;
  if (!out_is_fp16 && q_is_fp16) {
    TORCH_CHECK(false, "float32 output requires float32 q");
  }

#if defined(__HIPCC__)
  if (out_is_fp16 && q_is_fp16) {
    hipLaunchKernelGGL(
        (decode_int8_kv_mla_kernel<half, half>),
        grid,
        block,
        shared_bytes,
        stream,
        reinterpret_cast<half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<int8_t*>(k_cache.data_ptr<int8_t>()),
        reinterpret_cast<int8_t*>(v_cache.data_ptr<int8_t>()),
        reinterpret_cast<half*>(k_scale.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v_scale.data_ptr<at::Half>()),
        reinterpret_cast<int32_t*>(kv_indptr.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(kv_indices.data_ptr<int32_t>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        static_cast<int32_t>(batch_size),
        static_cast<int32_t>(head_num),
        static_cast<int32_t>(kv_head_num),
        static_cast<int32_t>(qk_head_dim),
        static_cast<int32_t>(v_head_dim),
        static_cast<int32_t>(kv_group_size),
        static_cast<int32_t>(max_tokens),
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        v_cache.stride(0),
        v_cache.stride(1),
        k_scale.stride(0),
        k_scale.stride(1),
        v_scale.stride(0),
        v_scale.stride(1),
        output.stride(0),
        output.stride(1),
        static_cast<float>(sm_scale),
        static_cast<float>(logit_cap));
  } else if (out_is_fp16 && !q_is_fp16) {
    hipLaunchKernelGGL(
        (decode_int8_kv_mla_kernel<float, half>),
        grid,
        block,
        shared_bytes,
        stream,
        reinterpret_cast<float*>(q.data_ptr<float>()),
        reinterpret_cast<int8_t*>(k_cache.data_ptr<int8_t>()),
        reinterpret_cast<int8_t*>(v_cache.data_ptr<int8_t>()),
        reinterpret_cast<half*>(k_scale.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v_scale.data_ptr<at::Half>()),
        reinterpret_cast<int32_t*>(kv_indptr.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(kv_indices.data_ptr<int32_t>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        static_cast<int32_t>(batch_size),
        static_cast<int32_t>(head_num),
        static_cast<int32_t>(kv_head_num),
        static_cast<int32_t>(qk_head_dim),
        static_cast<int32_t>(v_head_dim),
        static_cast<int32_t>(kv_group_size),
        static_cast<int32_t>(max_tokens),
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        v_cache.stride(0),
        v_cache.stride(1),
        k_scale.stride(0),
        k_scale.stride(1),
        v_scale.stride(0),
        v_scale.stride(1),
        output.stride(0),
        output.stride(1),
        static_cast<float>(sm_scale),
        static_cast<float>(logit_cap));
  } else {
    hipLaunchKernelGGL(
        (decode_int8_kv_mla_kernel<float, float>),
        grid,
        block,
        shared_bytes,
        stream,
        reinterpret_cast<float*>(q.data_ptr<float>()),
        reinterpret_cast<int8_t*>(k_cache.data_ptr<int8_t>()),
        reinterpret_cast<int8_t*>(v_cache.data_ptr<int8_t>()),
        reinterpret_cast<half*>(k_scale.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v_scale.data_ptr<at::Half>()),
        reinterpret_cast<int32_t*>(kv_indptr.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(kv_indices.data_ptr<int32_t>()),
        reinterpret_cast<float*>(output.data_ptr<float>()),
        static_cast<int32_t>(batch_size),
        static_cast<int32_t>(head_num),
        static_cast<int32_t>(kv_head_num),
        static_cast<int32_t>(qk_head_dim),
        static_cast<int32_t>(v_head_dim),
        static_cast<int32_t>(kv_group_size),
        static_cast<int32_t>(max_tokens),
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        v_cache.stride(0),
        v_cache.stride(1),
        k_scale.stride(0),
        k_scale.stride(1),
        v_scale.stride(0),
        v_scale.stride(1),
        output.stride(0),
        output.stride(1),
        static_cast<float>(sm_scale),
        static_cast<float>(logit_cap));
  }
#else
  if (out_is_fp16 && q_is_fp16) {
    decode_int8_kv_mla_kernel<half, half><<<grid, block, shared_bytes, stream>>>(
        reinterpret_cast<half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<int8_t*>(k_cache.data_ptr<int8_t>()),
        reinterpret_cast<int8_t*>(v_cache.data_ptr<int8_t>()),
        reinterpret_cast<half*>(k_scale.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v_scale.data_ptr<at::Half>()),
        reinterpret_cast<int32_t*>(kv_indptr.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(kv_indices.data_ptr<int32_t>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        static_cast<int32_t>(batch_size),
        static_cast<int32_t>(head_num),
        static_cast<int32_t>(kv_head_num),
        static_cast<int32_t>(qk_head_dim),
        static_cast<int32_t>(v_head_dim),
        static_cast<int32_t>(kv_group_size),
        static_cast<int32_t>(max_tokens),
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        v_cache.stride(0),
        v_cache.stride(1),
        k_scale.stride(0),
        k_scale.stride(1),
        v_scale.stride(0),
        v_scale.stride(1),
        output.stride(0),
        output.stride(1),
        static_cast<float>(sm_scale),
        static_cast<float>(logit_cap));
  } else if (out_is_fp16 && !q_is_fp16) {
    decode_int8_kv_mla_kernel<float, half><<<grid, block, shared_bytes, stream>>>(
        reinterpret_cast<float*>(q.data_ptr<float>()),
        reinterpret_cast<int8_t*>(k_cache.data_ptr<int8_t>()),
        reinterpret_cast<int8_t*>(v_cache.data_ptr<int8_t>()),
        reinterpret_cast<half*>(k_scale.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v_scale.data_ptr<at::Half>()),
        reinterpret_cast<int32_t*>(kv_indptr.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(kv_indices.data_ptr<int32_t>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        static_cast<int32_t>(batch_size),
        static_cast<int32_t>(head_num),
        static_cast<int32_t>(kv_head_num),
        static_cast<int32_t>(qk_head_dim),
        static_cast<int32_t>(v_head_dim),
        static_cast<int32_t>(kv_group_size),
        static_cast<int32_t>(max_tokens),
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        v_cache.stride(0),
        v_cache.stride(1),
        k_scale.stride(0),
        k_scale.stride(1),
        v_scale.stride(0),
        v_scale.stride(1),
        output.stride(0),
        output.stride(1),
        static_cast<float>(sm_scale),
        static_cast<float>(logit_cap));
  } else {
    decode_int8_kv_mla_kernel<float, float><<<grid, block, shared_bytes, stream>>>(
        reinterpret_cast<float*>(q.data_ptr<float>()),
        reinterpret_cast<int8_t*>(k_cache.data_ptr<int8_t>()),
        reinterpret_cast<int8_t*>(v_cache.data_ptr<int8_t>()),
        reinterpret_cast<half*>(k_scale.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v_scale.data_ptr<at::Half>()),
        reinterpret_cast<int32_t*>(kv_indptr.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(kv_indices.data_ptr<int32_t>()),
        reinterpret_cast<float*>(output.data_ptr<float>()),
        static_cast<int32_t>(batch_size),
        static_cast<int32_t>(head_num),
        static_cast<int32_t>(kv_head_num),
        static_cast<int32_t>(qk_head_dim),
        static_cast<int32_t>(v_head_dim),
        static_cast<int32_t>(kv_group_size),
        static_cast<int32_t>(max_tokens),
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        v_cache.stride(0),
        v_cache.stride(1),
        k_scale.stride(0),
        k_scale.stride(1),
        v_scale.stride(0),
        v_scale.stride(1),
        output.stride(0),
        output.stride(1),
        static_cast<float>(sm_scale),
        static_cast<float>(logit_cap));
  }
#endif
}
