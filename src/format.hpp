#pragma once

#include <string>
#include <vector>

namespace brrr {

enum class DType { fp64, fp32, tf32, fp16, bf16, e4m3, e2m1 };

enum class BlockScale { none, ue8m0, ue4m3 };

struct FormatSpec {
  std::string name;
  DType type;  // both inputs
  BlockScale scale;
  DType out;
};

// Every format brrr runs.
const std::vector<FormatSpec>& formats();

std::string to_string(DType t);
int bits(DType t);
int block_size(BlockScale s);  // values per scale: 32, 16, or 0 for none

}  // namespace brrr
