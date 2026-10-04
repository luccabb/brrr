#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace brrr {

// What NVML saw on one GPU while it burned one format.
struct FormatTelemetry {
  int samples = 0;
  unsigned max_temp_c = 0;
  unsigned last_temp_c = 0;
  double max_power_w = 0;
  unsigned min_sm_mhz = 0;    // lowest SM clock seen while busy
  uint64_t throttle = 0;      // OR of NVML clocks-event reasons seen
};

// Counters read at the start and end of the run, per GPU.
struct GpuTelemetry {
  double power_limit_w = 0;
  unsigned max_sm_mhz = 0;
  long long ecc_corrected = 0;    // increase during the run
  long long ecc_uncorrected = 0;
  long long remapped_rows = 0;    // pending + failed row remaps added during the run
};

// Samples GPUs through NVML (loaded with dlopen, so brrr has no driver link dependency).
// Every method is a no-op when NVML is unavailable.
class Monitor {
 public:
  Monitor();
  ~Monitor();
  void add_gpu(int gpu, const std::string& pci_bus_id);
  // Records one sample for `gpu`, attributed to `format` (skipped when format is empty).
  void sample(int gpu, const std::string& format);
  // Reads end-of-run counters; call once after the workers exit.
  void finish();

  const FormatTelemetry* format_stats(int gpu, const std::string& format) const;
  const GpuTelemetry* gpu_stats(int gpu) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Human-readable names for the throttle bits that explain slowness, e.g. "power cap".
std::vector<std::string> throttle_names(uint64_t reasons);

}  // namespace brrr
