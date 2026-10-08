#include "kernels.cuh"

#include "check.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace brrr {
namespace {

__device__ __forceinline__ float uniform(uint64_t seed, uint64_t i) {
  uint64_t x = seed + 0x9E3779B97F4A7C15ull * (i + 1);  // splitmix64
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  x ^= x >> 31;
  return static_cast<float>(x >> 40) * (2.0f / 16777216.0f) - 1.0f;  // [-1, 1)
}

template <typename T, typename Convert>
__global__ void fill_kernel(T* out, size_t count, uint64_t seed, Convert convert) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x) {
    out[i] = convert(uniform(seed, i));
  }
}

__global__ void random_bytes_kernel(uint8_t* out, size_t bytes, uint64_t seed, int lo, int span) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < bytes;
       i += size_t(gridDim.x) * blockDim.x) {
    out[i] = static_cast<uint8_t>(lo + int((uniform(seed, i) + 1.0f) * 0.5f * span) % span);
  }
}

__global__ void mismatch_kernel(const uint4* a, const uint4* b, size_t words,
                                const uint8_t* a_tail, const uint8_t* b_tail, size_t tail,
                                unsigned long long* count) {
  unsigned long long local = 0;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < words;
       i += size_t(gridDim.x) * blockDim.x) {
    const uint4 x = a[i], y = b[i];
    const uint32_t d[4] = {x.x ^ y.x, x.y ^ y.y, x.z ^ y.z, x.w ^ y.w};
    for (uint32_t w : d) {
      for (int byte = 0; byte < 4; ++byte) local += ((w >> (8 * byte)) & 0xFF) != 0;
    }
  }
  if (blockIdx.x == 0 && threadIdx.x < tail) local += a_tail[threadIdx.x] != b_tail[threadIdx.x];
  if (local) atomicAdd(count, local);
}

__global__ void corrupt_kernel(uint8_t* data, size_t offset) { data[offset] ^= 0x01; }

constexpr int kThreads = 256;
constexpr int kBlocks = 1024;

template <typename T, typename Convert>
void launch_fill(void* data, size_t count, uint64_t seed, cudaStream_t stream, Convert convert) {
  fill_kernel<<<kBlocks, kThreads, 0, stream>>>(static_cast<T*>(data), count, seed, convert);
}

}  // namespace

void fill_random(void* data, DType type, size_t count, uint64_t seed, cudaStream_t stream) {
  switch (type) {
    case DType::fp64:
      launch_fill<double>(data, count, seed, stream, [] __device__(float v) { return double(v); });
      break;
    case DType::fp32:
    case DType::tf32:
      launch_fill<float>(data, count, seed, stream, [] __device__(float v) { return v; });
      break;
    case DType::fp16:
      launch_fill<__half>(data, count, seed, stream, [] __device__(float v) { return __float2half(v); });
      break;
    case DType::bf16:
      launch_fill<__nv_bfloat16>(data, count, seed, stream,
                                 [] __device__(float v) { return __float2bfloat16(v); });
      break;
    case DType::e4m3:
      launch_fill<__nv_fp8_e4m3>(data, count, seed, stream,
                                 [] __device__(float v) { return __nv_fp8_e4m3(v); });
      break;
    case DType::e2m1:  // two values per byte; every nibble is a finite e2m1 value
      random_bytes_kernel<<<kBlocks, kThreads, 0, stream>>>(static_cast<uint8_t*>(data),
                                                            (count + 1) / 2, seed, 0, 256);
      break;
  }
  check_cuda(cudaGetLastError(), "fill kernel");
}

void fill_scales(void* data, BlockScale type, size_t count, uint64_t seed, cudaStream_t stream) {
  switch (type) {
    case BlockScale::ue8m0:  // biased exponent 126..128: 2^-1 .. 2^1
      random_bytes_kernel<<<kBlocks, kThreads, 0, stream>>>(static_cast<uint8_t*>(data), count,
                                                            seed, 126, 3);
      break;
    case BlockScale::ue4m3:
      launch_fill<__nv_fp8_e4m3>(data, count, seed, stream,
                                 [] __device__(float v) { return __nv_fp8_e4m3(1.25f + 0.75f * v); });
      break;
    case BlockScale::none:
      break;
  }
  check_cuda(cudaGetLastError(), "scale fill kernel");
}

void count_mismatches(const void* a, const void* b, size_t bytes,
                      unsigned long long* device_count, cudaStream_t stream) {
  const size_t words = bytes / sizeof(uint4);
  const size_t tail = bytes % sizeof(uint4);
  const auto* a8 = static_cast<const uint8_t*>(a) + words * sizeof(uint4);
  const auto* b8 = static_cast<const uint8_t*>(b) + words * sizeof(uint4);
  mismatch_kernel<<<kBlocks, kThreads, 0, stream>>>(static_cast<const uint4*>(a),
                                                    static_cast<const uint4*>(b), words, a8, b8,
                                                    tail, device_count);
  // A build without this GPU's arch fails here; unchecked, the burn would multiply zeros and never count an error.
  check_cuda(cudaGetLastError(), "mismatch kernel");
}

void corrupt_one_byte(void* data, size_t offset, cudaStream_t stream) {
  corrupt_kernel<<<1, 1, 0, stream>>>(static_cast<uint8_t*>(data), offset);
  check_cuda(cudaGetLastError(), "corrupt kernel");
}

}  // namespace brrr
