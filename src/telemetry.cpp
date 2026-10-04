#include "telemetry.hpp"

#include <dlfcn.h>
#include <nvml.h>

#include <algorithm>

namespace brrr {

constexpr unsigned long long kGpuIdle = 0x01;  // NVML clocks-event bit: GPU idle

struct Monitor::Impl {
  void* lib = nullptr;
  nvmlReturn_t (*init)() = nullptr;
  nvmlReturn_t (*shutdown)() = nullptr;
  nvmlReturn_t (*by_pci)(const char*, nvmlDevice_t*) = nullptr;
  nvmlReturn_t (*temperature)(nvmlDevice_t, nvmlTemperatureSensors_t, unsigned*) = nullptr;
  nvmlReturn_t (*power)(nvmlDevice_t, unsigned*) = nullptr;
  nvmlReturn_t (*power_limit)(nvmlDevice_t, unsigned*) = nullptr;
  nvmlReturn_t (*clock)(nvmlDevice_t, nvmlClockType_t, unsigned*) = nullptr;
  nvmlReturn_t (*max_clock)(nvmlDevice_t, nvmlClockType_t, unsigned*) = nullptr;
  nvmlReturn_t (*reasons)(nvmlDevice_t, unsigned long long*) = nullptr;
  nvmlReturn_t (*ecc)(nvmlDevice_t, nvmlMemoryErrorType_t, nvmlEccCounterType_t, unsigned long long*) = nullptr;
  nvmlReturn_t (*remapped)(nvmlDevice_t, unsigned*, unsigned*, unsigned*, unsigned*) = nullptr;

  struct Counters {
    long long ecc_corrected = -1, ecc_uncorrected = -1, remapped = -1;
  };
  std::map<int, nvmlDevice_t> devices;
  std::map<int, Counters> start;
  std::map<int, GpuTelemetry> gpus;
  std::map<std::pair<int, std::string>, FormatTelemetry> formats;

  template <typename F>
  void load(F& fn, const char* name) {
    fn = reinterpret_cast<F>(dlsym(lib, name));
  }

  Counters read_counters(nvmlDevice_t d) const {
    Counters c;
    unsigned long long n = 0;
    if (ecc && ecc(d, NVML_MEMORY_ERROR_TYPE_CORRECTED, NVML_VOLATILE_ECC, &n) == NVML_SUCCESS) c.ecc_corrected = n;
    if (ecc && ecc(d, NVML_MEMORY_ERROR_TYPE_UNCORRECTED, NVML_VOLATILE_ECC, &n) == NVML_SUCCESS) c.ecc_uncorrected = n;
    unsigned corr = 0, unc = 0, pending = 0, failed = 0;
    if (remapped && remapped(d, &corr, &unc, &pending, &failed) == NVML_SUCCESS) c.remapped = pending + failed;
    return c;
  }
};

Monitor::Monitor() : impl_(std::make_unique<Impl>()) {
  Impl& m = *impl_;
  m.lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!m.lib) return;
  m.load(m.init, "nvmlInit_v2");
  m.load(m.shutdown, "nvmlShutdown");
  m.load(m.by_pci, "nvmlDeviceGetHandleByPciBusId_v2");
  m.load(m.temperature, "nvmlDeviceGetTemperature");
  m.load(m.power, "nvmlDeviceGetPowerUsage");
  m.load(m.power_limit, "nvmlDeviceGetEnforcedPowerLimit");
  m.load(m.clock, "nvmlDeviceGetClockInfo");
  m.load(m.max_clock, "nvmlDeviceGetMaxClockInfo");
  m.load(m.reasons, "nvmlDeviceGetCurrentClocksEventReasons");
  m.load(m.ecc, "nvmlDeviceGetTotalEccErrors");
  m.load(m.remapped, "nvmlDeviceGetRemappedRows");
  if (!m.init || !m.by_pci || m.init() != NVML_SUCCESS) {
    dlclose(m.lib);
    m.lib = nullptr;
  }
}

Monitor::~Monitor() {
  if (impl_->lib) {
    impl_->shutdown();
    dlclose(impl_->lib);
  }
}

void Monitor::add_gpu(int gpu, const std::string& pci_bus_id) {
  Impl& m = *impl_;
  if (!m.lib) return;
  nvmlDevice_t d{};
  if (m.by_pci(pci_bus_id.c_str(), &d) != NVML_SUCCESS) return;
  m.devices[gpu] = d;
  m.start[gpu] = m.read_counters(d);
  GpuTelemetry& g = m.gpus[gpu];
  unsigned v = 0;
  if (m.power_limit && m.power_limit(d, &v) == NVML_SUCCESS) g.power_limit_w = v / 1000.0;
  if (m.max_clock && m.max_clock(d, NVML_CLOCK_SM, &v) == NVML_SUCCESS) g.max_sm_mhz = v;
}

void Monitor::sample(int gpu, const std::string& format) {
  Impl& m = *impl_;
  auto it = m.devices.find(gpu);
  if (!m.lib || it == m.devices.end() || format.empty()) return;
  const nvmlDevice_t d = it->second;
  FormatTelemetry& t = m.formats[{gpu, format}];
  unsigned v = 0;
  unsigned long long r = 0;
  if (m.temperature && m.temperature(d, NVML_TEMPERATURE_GPU, &v) == NVML_SUCCESS) {
    t.max_temp_c = std::max(t.max_temp_c, v);
    t.last_temp_c = v;
  }
  if (m.power && m.power(d, &v) == NVML_SUCCESS) t.max_power_w = std::max(t.max_power_w, v / 1000.0);
  const bool busy = !(m.reasons && m.reasons(d, &r) == NVML_SUCCESS && (r & kGpuIdle));
  if (m.reasons) t.throttle |= r & ~kGpuIdle;
  if (busy && m.clock && m.clock(d, NVML_CLOCK_SM, &v) == NVML_SUCCESS) {
    t.min_sm_mhz = t.samples == 0 || t.min_sm_mhz == 0 ? v : std::min(t.min_sm_mhz, v);
  }
  ++t.samples;
}

void Monitor::finish() {
  Impl& m = *impl_;
  for (const auto& [gpu, d] : m.devices) {
    const Impl::Counters end = m.read_counters(d), begin = m.start[gpu];
    GpuTelemetry& g = m.gpus[gpu];
    auto delta = [](long long a, long long b) { return a >= 0 && b >= 0 ? b - a : 0; };
    g.ecc_corrected = delta(begin.ecc_corrected, end.ecc_corrected);
    g.ecc_uncorrected = delta(begin.ecc_uncorrected, end.ecc_uncorrected);
    g.remapped_rows = delta(begin.remapped, end.remapped);
  }
}

const FormatTelemetry* Monitor::format_stats(int gpu, const std::string& format) const {
  auto it = impl_->formats.find({gpu, format});
  return it == impl_->formats.end() ? nullptr : &it->second;
}

const GpuTelemetry* Monitor::gpu_stats(int gpu) const {
  auto it = impl_->gpus.find(gpu);
  return it == impl_->gpus.end() ? nullptr : &it->second;
}

std::vector<std::string> throttle_names(uint64_t reasons) {
  // NVML clocks-event bits; the values are stable across NVML versions, the macro names are not.
  static const std::pair<uint64_t, const char*> kNames[] = {
      {0x04, "power cap"},    // SwPowerCap
      {0x08, "hw slowdown"},  // HwSlowdown
      {0x20, "thermal"},      // SwThermalSlowdown
      {0x40, "hw thermal"},   // HwThermalSlowdown
      {0x80, "power brake"},  // HwPowerBrakeSlowdown
      {0x02, "app clocks"},   // ApplicationsClocksSetting
  };
  std::vector<std::string> out;
  for (const auto& [bit, name] : kNames) {
    if (reasons & bit) out.push_back(name);
  }
  return out;
}

}  // namespace brrr
