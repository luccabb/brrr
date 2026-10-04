#include "format.hpp"

namespace brrr {

// Tensor-core paths are the tcgen05.mma kinds in NVIDIA's PTX ISA:
//   https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-kind-shapes
const std::vector<FormatSpec>& formats() {
  static const std::vector<FormatSpec> kFormats = {
      // 16-bit tensor-core MMA (kind::f16).
      {"bf16", DType::bf16, BlockScale::none, DType::bf16},
      // Same kind as bf16.
      {"fp16", DType::fp16, BlockScale::none, DType::fp16},
      // TF32 tensor-core MMA (kind::tf32).
      {"tf32", DType::tf32, BlockScale::none, DType::tf32},
      // FP32 FMA on the CUDA cores.
      {"fp32", DType::fp32, BlockScale::none, DType::fp32},
      // FP64 units.
      {"fp64", DType::fp64, BlockScale::none, DType::fp64},
      // FP8 tensor-core MMA with per-tensor scales (kind::f8f6f4).
      {"fp8", DType::e4m3, BlockScale::none, DType::bf16},
      // Same MMA as fp8.
      {"fp8-out", DType::e4m3, BlockScale::none, DType::e4m3},
      // Block-scaled FP8 (kind::mxf8f6f4).
      {"mxfp8", DType::e4m3, BlockScale::ue8m0, DType::bf16},
      // Block-scaled FP4 with E8M0 scales per 32 (kind::mxf4).
      {"mxfp4", DType::e2m1, BlockScale::ue8m0, DType::bf16},
      // Block-scaled FP4 with E4M3 scales per 16 (kind::mxf4nvf4).
      {"nvfp4", DType::e2m1, BlockScale::ue4m3, DType::bf16},
  };
  return kFormats;
}

std::string to_string(DType t) {
  switch (t) {
    case DType::fp64: return "fp64";
    case DType::fp32: return "fp32";
    case DType::tf32: return "tf32";
    case DType::fp16: return "fp16";
    case DType::bf16: return "bf16";
    case DType::e4m3: return "e4m3";
    case DType::e2m1: return "e2m1";
  }
  return "?";
}

int bits(DType t) {
  switch (t) {
    case DType::fp64: return 64;
    case DType::fp32: case DType::tf32: return 32;
    case DType::fp16: case DType::bf16: return 16;
    case DType::e4m3: return 8;
    case DType::e2m1: return 4;
  }
  return 0;
}

int block_size(BlockScale s) {
  switch (s) {
    case BlockScale::ue8m0: return 32;
    case BlockScale::ue4m3: return 16;
    case BlockScale::none: return 0;
  }
  return 0;
}

}  // namespace brrr
