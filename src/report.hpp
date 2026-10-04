#pragma once

#include <cstdio>
#include <vector>

#include "backend.hpp"
#include "telemetry.hpp"
#include "verdict.hpp"

namespace brrr {

struct RunSummary {
  std::vector<Device> devices;
  std::vector<FormatResult> results;  // judged
  double seconds_per_format = 0;
  const Monitor* monitor = nullptr;   // may be unavailable
  std::vector<Library> libraries;
};

// The text report: one row per GPU and format, then gpu-burn's "Tested N GPUs:" block.
void print_text(const RunSummary& run);

// The --json report.
void print_json(const RunSummary& run);

}  // namespace brrr
