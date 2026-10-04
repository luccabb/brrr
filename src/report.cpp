#include "report.hpp"

#include <cctype>
#include <string>

namespace brrr {
namespace {

std::string json_string(const std::string& s) {
  std::string out = "\"";
  for (const char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", c);
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out + "\"";
}

std::string json_array(const std::vector<std::string>& items) {
  std::string out = "[";
  for (size_t i = 0; i < items.size(); ++i) out += (i ? ", " : "") + json_string(items[i]);
  return out + "]";
}

std::string lower(const std::string& s) {
  std::string out = s;
  for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

std::string cell(const FormatResult& r) {
  if (r.verdict == Verdict::ok) return r.reason.empty() ? "OK" : "OK: " + r.reason;
  return to_string(r.verdict) + (r.reason.empty() ? "" : ": " + r.reason);
}

}  // namespace

void print_text(const RunSummary& run) {
  std::printf("%-4s %-12s %9s %9s %7s %5s %6s  %s\n", "gpu", "format", "TFLOPS", "vs peers", "errors",
              "maxT", "maxW", "verdict");
  for (const FormatResult& r : run.results) {
    if (r.verdict == Verdict::unsupported) {
      std::printf("%-4d %-12s %9s %9s %7s %5s %6s  %s\n", r.gpu, r.format.c_str(), "-", "-", "-", "-", "-",
                  cell(r).c_str());
      continue;
    }
    char temp[16] = "-", power[16] = "-";
    if (r.max_temp_c) std::snprintf(temp, sizeof temp, "%uC", r.max_temp_c);
    if (r.max_power_w > 0) std::snprintf(power, sizeof power, "%.0f", r.max_power_w);
    std::printf("%-4d %-12s %9.1f %+8.1f%% %7ld %5s %6s  %s\n", r.gpu, r.format.c_str(), r.tflops,
                r.vs_peers * 100, r.errors, temp, power, cell(r).c_str());
  }
  std::printf("\nTested %zu GPUs:\n", run.devices.size());
  for (const Device& d : run.devices) {
    std::printf("\tGPU %d: %s\n", d.index, to_string(gpu_verdict(run.results, d.index)).c_str());
  }
}

void print_json(const RunSummary& run) {
  std::printf("{\n  \"version\": 1,\n  \"seconds_per_format\": %.1f,\n  \"libraries\": [", run.seconds_per_format);
  for (size_t i = 0; i < run.libraries.size(); ++i) {
    const Library& l = run.libraries[i];
    std::printf("%s{\"name\": %s, \"version\": %s, \"path\": %s}", i ? ", " : "", json_string(l.name).c_str(),
                json_string(l.version).c_str(), json_string(l.path).c_str());
  }
  std::printf("],\n  \"gpus\": [");
  for (size_t i = 0; i < run.devices.size(); ++i) {
    const Device& d = run.devices[i];
    std::printf("%s\n    {\"index\": %d, \"name\": %s, \"pci_bus_id\": %s, \"verdict\": %s,\n     \"formats\": {",
                i ? "," : "", d.index, json_string(d.name).c_str(), json_string(d.pci_bus_id).c_str(),
                json_string(lower(to_string(gpu_verdict(run.results, d.index)))).c_str());
    bool first = true;
    for (const FormatResult& r : run.results) {
      if (r.gpu != d.index) continue;
      std::printf("%s\n       %s: {\"verdict\": %s, \"reason\": %s, \"tflops\": %.2f, \"errors\": %ld, "
                  "\"vs_peers\": %.4f, \"max_temp_c\": %u, \"max_power_w\": %.0f, "
                  "\"min_sm_mhz\": %u, \"conditions\": %s, \"backend\": %s}",
                  first ? "" : ",", json_string(r.format).c_str(), json_string(lower(to_string(r.verdict))).c_str(),
                  json_string(r.reason).c_str(), r.tflops, r.errors, r.vs_peers,
                  r.max_temp_c, r.max_power_w,
                  r.min_sm_mhz, json_array(r.conditions).c_str(),
                  r.backend.empty() ? "null" : json_string(r.backend).c_str());
      first = false;
    }
    std::printf("},\n     \"telemetry\": ");
    const GpuTelemetry* g = run.monitor ? run.monitor->gpu_stats(d.index) : nullptr;
    if (g) {
      std::printf("{\"power_limit_w\": %.0f, \"max_sm_mhz\": %u, \"ecc_corrected\": %lld, "
                  "\"ecc_uncorrected\": %lld, \"remapped_rows\": %lld}}",
                  g->power_limit_w, g->max_sm_mhz, g->ecc_corrected, g->ecc_uncorrected, g->remapped_rows);
    } else {
      std::printf("null}");
    }
  }
  std::printf("\n  ]\n}\n");
}

}  // namespace brrr
