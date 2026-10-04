#include "verdict.hpp"

#include <algorithm>
#include <cstdio>
#include <map>
#include <utility>

namespace brrr {
namespace {

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
}

std::string percent(double fraction) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%+.0f%%", fraction * 100);
  return buf;
}

}  // namespace

void judge(std::vector<FormatResult>& results) {
  std::map<std::pair<std::string, std::string>, std::vector<double>> peers;
  for (const auto& r : results) {
    if (r.ran && r.errors == 0 && r.failure.empty()) peers[{r.model, r.format}].push_back(r.tflops);
  }
  for (auto& r : results) {
    if (!r.ran) {
      r.verdict = Verdict::unsupported;
      r.reason = r.unsupported_reason;
      continue;
    }
    if (!r.failure.empty()) {
      r.verdict = Verdict::faulty;
      r.reason = r.failure;
      continue;
    }
    if (r.errors > 0) {
      r.verdict = Verdict::faulty;
      r.reason = std::to_string(r.errors) + (r.errors == 1 ? " mismatching byte" : " mismatching bytes");
      continue;
    }
    const auto& group = peers[{r.model, r.format}];
    const double peer_median = median(group);
    r.vs_peers = peer_median > 0 ? r.tflops / peer_median - 1 : 0;

    r.verdict = Verdict::ok;
    if (group.size() > 1 && r.vs_peers < -kPeerSlow) {
      r.verdict = Verdict::slow;
      r.reason = percent(r.vs_peers) + " vs peers";
    }
    if (r.verdict == Verdict::slow && !r.conditions.empty()) {
      std::string why;
      for (const std::string& c : r.conditions) why += (why.empty() ? "" : ", ") + c;
      r.reason += " (" + why + ")";
    }
  }
}

void confirm_slow(std::vector<FormatResult>& results, const std::vector<FormatResult>& rerun) {
  for (FormatResult& r : results) {
    if (r.verdict != Verdict::slow) continue;
    for (const FormatResult& again : rerun) {
      if (again.gpu != r.gpu || again.format != r.format) continue;
      if (again.verdict == Verdict::slow) {
        r.reason += "; again on rerun: " + again.reason;
      } else if (again.verdict == Verdict::faulty) {
        r.verdict = Verdict::faulty;
        r.reason = again.reason;
      } else {
        r.verdict = Verdict::ok;
        r.reason = "slow once (" + r.reason + "), not on rerun";
      }
    }
  }
}

Verdict gpu_verdict(const std::vector<FormatResult>& results, int gpu) {
  Verdict worst = Verdict::ok;
  for (const auto& r : results) {
    if (r.gpu != gpu || r.verdict == Verdict::unsupported) continue;
    if (r.verdict == Verdict::faulty) return Verdict::faulty;
    if (r.verdict == Verdict::slow) worst = Verdict::slow;
  }
  return worst;
}

std::string to_string(Verdict v) {
  switch (v) {
    case Verdict::ok: return "OK";
    case Verdict::slow: return "SLOW";
    case Verdict::faulty: return "FAULTY";
    case Verdict::unsupported: return "UNSUPPORTED";
  }
  return "?";
}

}  // namespace brrr
