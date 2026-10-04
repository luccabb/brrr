#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "format.hpp"

namespace brrr {

// Fills `count` elements of `type` with values in [-1, 1) derived from (seed, index), so
// every GPU given the same seed holds the same inputs.
void fill_random(void* data, DType type, size_t count, uint64_t seed, cudaStream_t stream);

// Fills `count` block-scale factors of `type` with values in [0.5, 2] (powers of two for
// ue8m0), so scaled products stay finite while still exercising the scale hardware.
void fill_scales(void* data, BlockScale type, size_t count, uint64_t seed, cudaStream_t stream);

// Adds the number of differing bytes between `a` and `b` to `*device_count`.
void count_mismatches(const void* a, const void* b, size_t bytes,
                      unsigned long long* device_count, cudaStream_t stream);

// Overwrites one byte of `data` (fault-injection for tests; never used in normal runs).
void corrupt_one_byte(void* data, size_t offset, cudaStream_t stream);

}  // namespace brrr
