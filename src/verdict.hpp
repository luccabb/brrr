#pragma once

#include <string>
#include <vector>

namespace brrr {

enum class Verdict { ok, slow, faulty, unsupported };

struct FormatResult {
  int gpu = 0;
  std::string model;   // GPU name; peers are GPUs with the same model
  std::string format;
  bool ran = false;    // false when the format was unsupported on this GPU
  std::string unsupported_reason;
  double tflops = 0;
  long errors = 0;     // mismatching bytes summed over all repeats
  std::string failure; // set when the worker failed or stopped mid-format: FAULTY
  std::string backend;  // the library that ran it
  // Telemetry for this GPU and format; `conditions` explain a SLOW verdict.
  unsigned max_temp_c = 0;
  double max_power_w = 0;
  unsigned min_sm_mhz = 0;
  std::vector<std::string> conditions;
  // Filled by judge():
  double vs_peers = 0;     // fraction relative to the peer median, e.g. -0.12
  Verdict verdict = Verdict::ok;
  std::string reason;
};

// SLOW when this far below the median of the same GPU model's results.
constexpr double kPeerSlow = 0.08;

// Sets vs_peers, verdict and reason on every result.
void judge(std::vector<FormatResult>& results);

// Keeps SLOW only where a second, judged run (`rerun`, same GPUs and format) is slow too;
// a FAULTY rerun overrides. Other results are left as they are.
void confirm_slow(std::vector<FormatResult>& results, const std::vector<FormatResult>& rerun);

// Worst verdict over a GPU's formats: faulty > slow > ok. Unsupported formats are ignored.
Verdict gpu_verdict(const std::vector<FormatResult>& results, int gpu);

std::string to_string(Verdict v);  // OK, SLOW, FAULTY, UNSUPPORTED

}  // namespace brrr
