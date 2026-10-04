#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "format.hpp"

namespace brrr {

// Fixed-size message from a worker to the parent, written atomically to a pipe.
struct Record {
  enum class Kind : int32_t { progress, done, unsupported, failed };
  Kind kind = Kind::progress;
  int32_t gpu = 0;
  int32_t format = 0;    // index into the run's format list
  int64_t iterations = 0;
  float slice_done = 0;  // how far into this format's seconds, 0..1
  int64_t errors = 0;    // mismatching bytes so far
  double tflops = 0;     // steady state, after warm-up (so far, on progress records)
  int64_t memory_bytes = 0;  // reference plus the ring of outputs
  int64_t free_bytes = 0;    // free before brrr allocated anything for the format
  int64_t total_bytes = 0;
  char backend[16] = {};     // the library that ran it
  char reason[200] = {};
};

struct WorkerPlan {
  int gpu = 0;
  std::vector<FormatSpec> formats;
  double seconds_per_format = 10;
  uint64_t seed = 0x6272727221ull;  // same on every GPU, so peers multiply the same inputs
  long inject_fault_after = -1;     // test hook: corrupt one output after N iterations
  double memory_fraction = 0.9;     // -m N%: share of free memory for outputs, like gpu-burn
  size_t memory_bytes = 0;          // -m N: that many bytes instead, when set
};

// Runs in a child process: burns each format for its slice and reports to `fd`.
int run_worker(const WorkerPlan& plan, int fd);

}  // namespace brrr
