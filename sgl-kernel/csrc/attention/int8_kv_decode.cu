#include <torch/extension.h>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#else
#include <cuda_runtime.h>
#endif

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

constexpr int kWarpSize = 32;

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

__device__ __forceinline__ float warp_reduce_max(float val) {
#if defined(__HIPCC__)
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_down(val, offset, kWarpSize));
  }
#else
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
#endif
  return val;
}

__global__ void decode_int8_kv_kernel(
    const half* __restrict__ q,
    const int8_t* __restrict__ k_cache,
    const int8_t* __restrict__ v_cache,
    const half* __restrict__ k_scale,
    const half* __restrict__ v_scale,
    const int32_t* __restrict__ kv_indptr,
    const int32_t* __restrict__ kv_indices,
    half* __restrict__ out,
    int32_t batch_size,
    int32_t head_num,
    int32_t kv_head_num,
    int32_t qk_head_dim,
    int32_t v_head_dim,
    int32_t kv_group_size,
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

  int lane = threadIdx.x;
  if (lane >= kWarpSize) {
    return;
  }

  extern __shared__ char smem[];
  half* q_shared = reinterpret_cast<half*>(smem);
  float* acc_shared = reinterpret_cast<float*>(q_shared + qk_head_dim);

  // Load q into shared memory.
  for (int d = lane; d < qk_head_dim; d += kWarpSize) {
    q_shared[d] = q[b * stride_qbs + h * stride_qh + d];
  }

  if (lane == 0) {
    for (int d = 0; d < v_head_dim; ++d) {
      acc_shared[d] = 0.0f;
    }
  }
#if defined(__HIPCC__)
  __syncthreads();
#else
  __syncthreads();
#endif

  int32_t kv_start = kv_indptr[b];
  int32_t kv_end = kv_indptr[b + 1];

  float m = -INFINITY;
  float l = 0.0f;

  for (int32_t start = kv_start; start < kv_end; start += kWarpSize) {
    int32_t kv_idx = start + lane;
    bool valid = kv_idx < kv_end;
    int32_t token_idx = valid ? kv_indices[kv_idx] : 0;

    float qk = -INFINITY;
    if (valid) {
      float sum = 0.0f;
      for (int g = 0; g < qk_head_dim; g += kv_group_size) {
        int scale_idx = g / kv_group_size;
        float scale = __half2float(
            k_scale[token_idx * stride_ks + kv_head * stride_ksh + scale_idx]);
        for (int d = 0; d < kv_group_size && (g + d) < qk_head_dim; ++d) {
          int idx = g + d;
          int8_t k_val = k_cache[token_idx * stride_kbs + kv_head * stride_kh + idx];
          float kf = static_cast<float>(k_val) * scale;
          float qf = __half2float(q_shared[idx]);
          sum += qf * kf;
        }
      }
      qk = sum * sm_scale;
      if (logit_cap > 0.0f) {
        qk = logit_cap * tanhf(qk / logit_cap);
      }
    }

    float block_max = warp_reduce_max(qk);
    float m_new = fmaxf(m, block_max);

    float exp_scale = (m == -INFINITY) ? 0.0f : expf(m - m_new);
    float p = valid ? expf(qk - m_new) : 0.0f;
    float p_sum = warp_reduce_sum(p);
    float l_new = l * exp_scale + p_sum;

    if (lane == 0 && exp_scale != 1.0f) {
      for (int d = 0; d < v_head_dim; ++d) {
        acc_shared[d] *= exp_scale;
      }
    }
#if defined(__HIPCC__)
    __syncthreads();
#else
    __syncthreads();
#endif

    if (valid) {
      for (int g = 0; g < v_head_dim; g += kv_group_size) {
        int scale_idx = g / kv_group_size;
        float scale = __half2float(
            v_scale[token_idx * stride_vs + kv_head * stride_vsh + scale_idx]);
        for (int d = 0; d < kv_group_size && (g + d) < v_head_dim; ++d) {
          int idx = g + d;
          int8_t v_val = v_cache[token_idx * stride_vbs + kv_head * stride_vh + idx];
          float contrib = p * (static_cast<float>(v_val) * scale);
          float sum = warp_reduce_sum(contrib);
          if (lane == 0) {
            acc_shared[idx] += sum;
          }
#if defined(__HIPCC__)
          __syncthreads();
#else
          __syncthreads();
#endif
        }
      }
    }

    m = m_new;
    l = l_new;
#if defined(__HIPCC__)
    __syncthreads();
#else
    __syncthreads();
#endif
  }

  if (lane == 0) {
    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int d = 0; d < v_head_dim; ++d) {
      float out_val = acc_shared[d] * inv_l;
      out[b * stride_obs + h * stride_oh + d] = __float2half(out_val);
    }
  }
}

}  // namespace

void decode_attention_int8_kv(
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
  const auto v_head_dim = v_cache.size(2);
  const auto kv_head_num = k_cache.size(1);

  TORCH_CHECK(q.is_cuda(), "q must be CUDA/HIP");
  TORCH_CHECK(k_cache.is_cuda(), "k_cache must be CUDA/HIP");
  TORCH_CHECK(v_cache.is_cuda(), "v_cache must be CUDA/HIP");
  TORCH_CHECK(k_scale.is_cuda(), "k_scale must be CUDA/HIP");
  TORCH_CHECK(v_scale.is_cuda(), "v_scale must be CUDA/HIP");
  TORCH_CHECK(output.is_cuda(), "output must be CUDA/HIP");
  TORCH_CHECK(kv_indptr.is_cuda(), "kv_indptr must be CUDA/HIP");
  TORCH_CHECK(kv_indices.is_cuda(), "kv_indices must be CUDA/HIP");
  TORCH_CHECK(q.scalar_type() == at::kHalf, "q must be float16");
  TORCH_CHECK(k_cache.scalar_type() == at::kChar, "k_cache must be int8");
  TORCH_CHECK(v_cache.scalar_type() == at::kChar, "v_cache must be int8");
  TORCH_CHECK(k_scale.scalar_type() == at::kHalf, "k_scale must be float16");
  TORCH_CHECK(v_scale.scalar_type() == at::kHalf, "v_scale must be float16");
  TORCH_CHECK(output.scalar_type() == at::kHalf, "output must be float16");
  TORCH_CHECK(kv_indptr.scalar_type() == at::kInt, "kv_indptr must be int32");
  TORCH_CHECK(kv_indices.scalar_type() == at::kInt, "kv_indices must be int32");
  TORCH_CHECK(
      qk_head_dim % kv_group_size == 0,
      "qk_head_dim must be divisible by kv_group_size");
  TORCH_CHECK(
      v_head_dim % kv_group_size == 0,
      "v_head_dim must be divisible by kv_group_size");

  const dim3 grid(batch_size, head_num, 1);
  const dim3 block(kWarpSize, 1, 1);
  size_t shared_bytes =
      sizeof(half) * qk_head_dim + sizeof(float) * v_head_dim;

  const auto stream = at::cuda::getDefaultCUDAStream();

#if defined(__HIPCC__)
  hipLaunchKernelGGL(
      decode_int8_kv_kernel,
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
#else
  decode_int8_kv_kernel<<<grid, block, shared_bytes, stream>>>(
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
#endif
}
