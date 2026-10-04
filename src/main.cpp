// brrr: burn every GPU in the formats models run in, check every result bit for bit,
// and flag GPUs that are wrong or slow. See README.md.

#include <cuda_runtime.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend.hpp"
#include "format.hpp"
#include "report.hpp"
#include "telemetry.hpp"
#include "verdict.hpp"
#include "worker.hpp"

namespace brrr {
namespace {

constexpr int kExitOk = 0, kExitSlow = 1, kExitFaulty = 2, kExitCannotRun = 3;

struct Options {
  double seconds = 10;
  bool per_format = false;
  bool list = false;
  bool json = false;
  std::optional<int> gpu;
  std::vector<std::string> formats;  // empty: bf16 (every format with -l); "all": every format
  long inject_fault_after = -1;
  double memory_fraction = 0.9;  // -m N%
  size_t memory_bytes = 0;       // -m N (MiB)
};

void usage() {
  std::puts(
      "Usage: brrr [options] [seconds]       (default 10 s, split across formats)\n"
      "  -f, --formats LIST     comma-separated formats, or all (default: bf16)\n"
      "  -i, --gpu INDEX        one GPU (default: all)\n"
      "  -m N|N%                GPU memory to use: N MiB or N% of free (default 90%)\n"
      "  -l, --list             list GPUs and the formats each can run\n"
      "      --json             machine-readable result on stdout\n"
      "      --per-format       seconds apply to each format, not the total\n"
      "  -d                     same as -f fp64\n"
      "  -tc                    same as -f tf32\n"
      "  -h, --help\n"
      "Exit: 0 OK, 1 slow, 2 faulty, 3 could not run.");
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  for (std::string item; std::getline(ss, item, ',');) {
    if (!item.empty()) out.push_back(item);
  }
  return out;
}

Options parse_args(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "brrr: %s needs a value\n", a.c_str());
        std::exit(kExitCannotRun);
      }
      return argv[++i];
    };
    if (a == "-h" || a == "--help") { usage(); std::exit(kExitOk); }
    else if (a == "-f" || a == "--formats") { o.formats = split(value()); }
    else if (a == "-i" || a == "--gpu") { o.gpu = std::stoi(value()); }
    else if (a == "-l" || a == "--list") { o.list = true; }
    else if (a == "--json") { o.json = true; }
    else if (a == "--per-format") { o.per_format = true; }
    else if (a == "-m") {
      const std::string v = value();
      if (!v.empty() && v.back() == '%') o.memory_fraction = std::stod(v) / 100;
      else o.memory_bytes = std::stoull(v) << 20;
    }
    else if (a == "-d") { o.formats = {"fp64"}; }
    else if (a == "-tc") { o.formats = {"tf32"}; }
    else if (a == "--inject-fault-after") { o.inject_fault_after = std::stol(value()); }  // tests only
    else if (!a.empty() && a[0] != '-') { o.seconds = std::stod(a); }
    else {
      std::fprintf(stderr, "brrr: unknown option %s\n", a.c_str());
      usage();
      std::exit(kExitCannotRun);
    }
  }
  return o;
}

// A process that forks CUDA-using children must not use CUDA itself, so discovery (and
// -l probing) runs in a short-lived child that reports over a pipe.
std::vector<Device> discover(const std::vector<FormatSpec>& probe_formats, bool print_support,
                             std::optional<int> only_gpu) {
  int fds[2];
  if (pipe(fds) != 0) return {};
  std::fflush(nullptr);
  const pid_t pid = fork();
  if (pid == 0) {
    close(fds[0]);
    FILE* out = fdopen(fds[1], "w");
    int count = 0;
    if (const cudaError_t e = cudaGetDeviceCount(&count); e != cudaSuccess) {
      count = 0;
      int driver = 0;
      cudaDriverGetVersion(&driver);
      std::fprintf(out, "err\t%s", cudaGetErrorString(e));
      if (e == cudaErrorInsufficientDriver) {
        std::fprintf(out, " (the driver supports CUDA %d.%d; brrr needs 13.0, driver 580 or newer)", driver / 1000,
                     driver % 1000 / 10);
      }
      std::fprintf(out, "\n");
    }
    for (int g = 0; g < count; ++g) {
      if (only_gpu && g != *only_gpu) continue;
      cudaDeviceProp p{};
      cudaGetDeviceProperties(&p, g);
      char pci[32] = {};
      cudaDeviceGetPCIBusId(pci, sizeof pci, g);
      std::fprintf(out, "gpu\t%d\t%d\t%zu\t%s\t%s\n", g, p.major * 10 + p.minor, p.totalGlobalMem, pci, p.name);
      if (!print_support) continue;
      cudaSetDevice(g);
      for (const FormatSpec& f : probe_formats) {
        Support s;
        const std::unique_ptr<Backend> backend = pick_backend(f, s);
        const std::string status = !backend ? s.reason
                                   : std::strcmp(backend->name(), "cuBLASLt") == 0 ? "ok"
                                   : std::string("ok via ") + backend->name();
        std::fprintf(out, "fmt\t%d\t%s\t%s\n", g, f.name.c_str(), status.c_str());
      }
    }
    std::fclose(out);
    _exit(0);
  }
  close(fds[1]);
  std::vector<Device> devices;
  FILE* in = fdopen(fds[0], "r");
  char line[1024];
  while (std::fgets(line, sizeof line, in)) {
    std::string l(line);
    if (!l.empty() && l.back() == '\n') l.pop_back();
    if (l.rfind("gpu\t", 0) == 0) {
      Device d;
      char pci[64] = {}, name[256] = {};
      std::sscanf(l.c_str(), "gpu\t%d\t%d\t%zu\t%63[^\t]\t%255[^\n]", &d.index, &d.arch, &d.memory_bytes, pci, name);
      d.pci_bus_id = pci;
      d.name = name;
      devices.push_back(d);
      if (print_support) {
        std::printf("GPU %d: %s (compute %d.%d, %zu MiB, %s)\n", d.index, d.name.c_str(), d.arch / 10,
                    d.arch % 10, d.memory_bytes >> 20, d.pci_bus_id.c_str());
      }
    } else if (l.rfind("err\t", 0) == 0) {
      std::fprintf(stderr, "brrr: %s\n", l.c_str() + 4);
    } else if (print_support && l.rfind("fmt\t", 0) == 0) {
      int g = 0;
      char fmt[128] = {}, status[600] = {};
      std::sscanf(l.c_str(), "fmt\t%d\t%127[^\t]\t%599[^\n]", &g, fmt, status);
      std::string shown = status;
      if (shown == "ok") shown = "supported";
      else if (shown.rfind("ok via ", 0) == 0) shown = "supported (" + shown.substr(7) + ")";
      std::printf("  %-14s %s\n", fmt, shown.c_str());
    }
  }
  std::fclose(in);
  waitpid(pid, nullptr, 0);
  return devices;
}

volatile sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }

struct Worker {
  int gpu = 0;
  pid_t pid = -1;
  int fd = -1;
  bool open = true;
  bool hung = false;  // killed after kHangSeconds without a record
};

// Workers report about once a second while burning; setup (loading cuBLASLt, autotuning)
// can take a minute. Silence this long means the GPU stopped responding.
constexpr double kHangSeconds = 300;

std::vector<Worker> start_workers(const std::vector<Device>& devices, const std::vector<FormatSpec>& formats,
                                  double slice, const Options& opt) {
  std::vector<Worker> workers;
  std::fflush(nullptr);  // children must not inherit (and later re-print) buffered output
  for (const Device& d : devices) {
    int fds[2];
    if (pipe(fds) != 0) throw std::runtime_error("pipe failed");
    const pid_t pid = fork();
    if (pid == 0) {
      close(fds[0]);
      WorkerPlan plan{d.index, formats, slice};
      plan.inject_fault_after = opt.inject_fault_after;
      plan.memory_fraction = opt.memory_fraction;
      plan.memory_bytes = opt.memory_bytes;
      _exit(run_worker(plan, fds[1]));
    }
    close(fds[1]);
    workers.push_back({d.index, pid, fds[0]});
  }
  return workers;
}

void stop_workers(std::vector<Worker>& workers, bool quiet) {
  if (!quiet) std::puts("\nKilling processes with SIGTERM (soft kill)");
  for (auto& w : workers) if (w.open) kill(w.pid, SIGTERM);
  sleep(2);
  if (!quiet) std::puts("Killing processes with SIGKILL (force kill)");
  for (auto& w : workers) if (w.open) kill(w.pid, SIGKILL);
}

using Latest = std::map<int, std::map<int, Record>>;  // gpu -> format index -> last record

// Reads worker records until every worker exits (or a signal), sampling telemetry once a
// second and printing gpu-burn style progress every 10% of the run.
void collect(std::vector<Worker>& workers, const std::vector<FormatSpec>& formats,
             Monitor& monitor, const std::string& pass, bool quiet, Latest& latest) {
  std::map<int, int> current;  // gpu -> format index it is burning
  std::set<int> memory_shown;
  const bool live = !quiet && isatty(STDOUT_FILENO);  // a terminal gets a line updated every second
  const auto t0 = std::chrono::steady_clock::now();
  double next_progress = 0.1;
  size_t open = workers.size();
  std::map<int, std::chrono::steady_clock::time_point> last_seen;
  for (const auto& w : workers) last_seen[w.gpu] = t0;
  while (open > 0 && !g_stop) {
    std::vector<pollfd> pfds;
    for (const auto& w : workers) if (w.open) pfds.push_back({w.fd, POLLIN, 0});
    poll(pfds.data(), pfds.size(), 1000);
    for (auto& w : workers) {
      if (!w.open) continue;
      pollfd p{w.fd, POLLIN, 0};
      while (poll(&p, 1, 0) > 0) {
        Record r;
        if (read(w.fd, &r, sizeof r) != sizeof r) {
          w.open = false;
          close(w.fd);
          --open;
          break;
        }
        if (!quiet && pass.empty() && r.memory_bytes > 0 && !memory_shown.count(r.gpu)) {
          memory_shown.insert(r.gpu);
          std::printf("%sGPU %d: using %.1f GiB of %.1f GiB free (%.1f GiB total)\n", live ? "\r\033[K" : "",
                      r.gpu, r.memory_bytes / double(1ull << 30), r.free_bytes / double(1ull << 30),
                      r.total_bytes / double(1ull << 30));
        }
        latest[r.gpu][r.format] = r;
        last_seen[w.gpu] = std::chrono::steady_clock::now();
        if (r.kind == Record::Kind::progress) current[r.gpu] = r.format;
        else current.erase(r.gpu);
      }
      const double silent = std::chrono::duration<double>(std::chrono::steady_clock::now() - last_seen[w.gpu]).count();
      if (w.open && silent > kHangSeconds) {
        kill(w.pid, SIGKILL);
        close(w.fd);
        w.open = false;
        w.hung = true;
        --open;
        // The format it was on: the one in progress, or the one after the last finished.
        const auto& seen = latest[w.gpu];
        const int idx = current.count(w.gpu) ? current[w.gpu] : seen.empty() ? 0 : seen.rbegin()->first + 1;
        current.erase(w.gpu);
        if (idx < int(formats.size())) {
          Record r = seen.count(idx) ? seen.at(idx) : Record{};
          r.kind = Record::Kind::failed;
          r.gpu = w.gpu;
          r.format = idx;
          std::snprintf(r.reason, sizeof r.reason, "hung: no progress for %.0f s", kHangSeconds);
          latest[w.gpu][idx] = r;
        }
      }
    }
    for (const auto& [gpu, idx] : current) monitor.sample(gpu, formats[idx].name + pass);

    // Progress is burning time, averaged over GPUs, so setup (loading libraries, tuning)
    // does not count; it starts once every GPU has reported.
    double done = 0;
    bool all_started = true;
    for (const auto& w : workers) {
      if (latest[w.gpu].empty()) all_started = false;
      for (const auto& [i, rec] : latest[w.gpu]) done += rec.kind == Record::Kind::progress ? rec.slice_done : 1;
    }
    done = std::min(done / (workers.size() * formats.size()), 1.0);
    if (!quiet && all_started) {
      std::string status;
      char buf[96];
      for (const auto& w : workers) {
        long errors = 0;
        int idx = -1;
        for (const auto& [i, rec] : latest[w.gpu]) {
          errors += rec.errors;
          idx = i;
        }
        const Record& r = latest[w.gpu][idx];
        const FormatTelemetry* t = monitor.format_stats(w.gpu, formats[idx].name + pass);
        if (r.tflops > 0 && t) {
          std::snprintf(buf, sizeof buf, " gpu%d %s %.0f TF %uC errors:%ld |", w.gpu, formats[idx].name.c_str(),
                        r.tflops, t->last_temp_c, errors);
        } else {
          std::snprintf(buf, sizeof buf, " gpu%d %s errors:%ld |", w.gpu, formats[idx].name.c_str(), errors);
        }
        status += buf;
      }
      const char* clear = live ? "\r\033[K" : "";
      for (; next_progress <= 1.0 + 1e-9 && done >= next_progress - 1e-9; next_progress += 0.1) {
        std::printf("%s%5.1f%% %s\n", clear, 100 * next_progress, status.c_str());
      }
      if (live) std::printf("%s%5.1f%% %s", clear, 100 * done, status.c_str());
      std::fflush(stdout);
    }
  }
}

void apply_telemetry(FormatResult& fr, const Monitor& monitor, const std::string& pass) {
  const FormatTelemetry* t = monitor.format_stats(fr.gpu, fr.format + pass);
  if (!t) return;
  fr.max_temp_c = t->max_temp_c;
  fr.max_power_w = t->max_power_w;
  fr.min_sm_mhz = t->min_sm_mhz;
  fr.conditions = throttle_names(t->throttle);
  const GpuTelemetry* g = monitor.gpu_stats(fr.gpu);
  if (g && g->max_sm_mhz && t->min_sm_mhz && t->min_sm_mhz < g->max_sm_mhz * 0.9) {
    fr.conditions.push_back("SM clock down to " + std::to_string(t->min_sm_mhz) + " MHz");
  }
}

std::vector<FormatResult> build_results(const std::vector<Device>& devices, const std::vector<FormatSpec>& formats,
                                        Latest& latest, const Monitor& monitor, const std::string& pass) {
  std::vector<FormatResult> results;
  for (const Device& d : devices) {
    for (size_t i = 0; i < formats.size(); ++i) {
      FormatResult fr;
      fr.gpu = d.index;
      fr.model = d.name;
      fr.format = formats[i].name;
      auto it = latest[d.index].find(static_cast<int>(i));
      if (it == latest[d.index].end()) {
        fr.unsupported_reason = g_stop ? "interrupted" : "worker exited before this format";
      } else {
        const Record& r = it->second;
        fr.ran = r.kind != Record::Kind::unsupported;
        fr.unsupported_reason = r.reason;
        fr.tflops = r.tflops;
        fr.errors = r.errors;
        fr.backend = r.backend;
        if (r.kind == Record::Kind::failed) fr.failure = r.reason;
        if (r.kind == Record::Kind::progress) fr.failure = g_stop ? "interrupted" : "worker stopped before finishing";
      }
      apply_telemetry(fr, monitor, pass);
      results.push_back(fr);
    }
  }
  return results;
}

// New uncorrectable ECC errors or row remaps during the run fail the GPU even when every
// result matched. Counters are only final after Monitor::finish().
void apply_memory_errors(std::vector<FormatResult>& results, const std::vector<Device>& devices,
                         const Monitor& monitor) {
  for (const Device& d : devices) {
    const GpuTelemetry* g = monitor.gpu_stats(d.index);
    if (!g || (g->ecc_uncorrected == 0 && g->remapped_rows == 0)) continue;
    for (auto r = results.rbegin(); r != results.rend(); ++r) {
      if (r->gpu != d.index || r->verdict == Verdict::unsupported) continue;
      r->verdict = Verdict::faulty;
      r->reason = std::to_string(g->ecc_uncorrected) + " uncorrectable ECC errors, " +
                  std::to_string(g->remapped_rows) + " row remaps during the run";
      break;
    }
  }
}

std::vector<FormatSpec> select_formats(const Options& opt) {
  if ((opt.list && opt.formats.empty()) || opt.formats == std::vector<std::string>{"all"}) return formats();
  std::vector<FormatSpec> selected;
  for (const std::string& name : opt.formats.empty() ? std::vector<std::string>{"bf16"} : opt.formats) {
    auto it = std::find_if(formats().begin(), formats().end(), [&](const FormatSpec& f) { return f.name == name; });
    if (it == formats().end()) throw std::runtime_error("unknown format '" + name + "' (see brrr -l)");
    selected.push_back(*it);
  }
  return selected;
}


int exit_code(const RunSummary& run) {
  int code = kExitOk;
  for (const Device& d : run.devices) {
    const Verdict v = gpu_verdict(run.results, d.index);
    if (v == Verdict::faulty) code = kExitFaulty;
    else if (v == Verdict::slow && code == kExitOk) code = kExitSlow;
  }
  for (const auto& r : run.results) {
    if (r.verdict != Verdict::unsupported) return code;
  }
  return kExitCannotRun;  // nothing ran
}

void print_libraries(const std::vector<Library>& libraries) {
  for (const Library& l : libraries) std::printf("%s %s: %s\n", l.name.c_str(), l.version.c_str(), l.path.c_str());
}

int run(const Options& opt) {
  const std::vector<FormatSpec> formats = select_formats(opt);
  std::vector<Library> libraries = {cublaslt_library()};
  if (const auto cudnn = cudnn_library()) libraries.push_back(*cudnn);
  if (opt.list) {
    print_libraries(libraries);
    discover(formats, /*print_support=*/true, opt.gpu);
    return kExitOk;
  }
  const std::vector<Device> devices = discover({}, false, opt.gpu);
  if (devices.empty()) {
    std::fprintf(stderr, "brrr: no CUDA GPUs found\n");
    return kExitCannotRun;
  }

  const double slice = opt.per_format ? opt.seconds : opt.seconds / formats.size();
  const double total = slice * formats.size();
  const bool quiet = opt.json;
  if (!quiet) {
    for (const Device& d : devices) {
      std::printf("GPU %d: %s (compute %d.%d)\n", d.index, d.name.c_str(), d.arch / 10, d.arch % 10);
    }
    print_libraries(libraries);
    std::printf("Burning for %.0f seconds:", total);
    for (const auto& f : formats) std::printf(" %s", f.name.c_str());
    if (formats.size() > 1) std::printf(" (%.1f s each)", slice);
    std::printf("\n");
  }
  std::fflush(stdout);

  Monitor monitor;
  for (const Device& d : devices) monitor.add_gpu(d.index, d.pci_bus_id);
  signal(SIGINT, on_signal);
  signal(SIGTERM, on_signal);
  auto burn = [&](const std::vector<FormatSpec>& pass_formats, const std::string& pass) {
    std::vector<Worker> workers = start_workers(devices, pass_formats, slice, opt);
    Latest latest;
    collect(workers, pass_formats, monitor, pass, quiet, latest);
    if (g_stop) stop_workers(workers, quiet);
    for (auto& w : workers) waitpid(w.pid, nullptr, w.hung ? WNOHANG : 0);  // a hung GPU can block exit
    std::vector<FormatResult> results = build_results(devices, pass_formats, latest, monitor, pass);
    judge(results);
    return results;
  };

  std::vector<FormatResult> results = burn(formats, "");
  // A single slice at the power cap can dip several percent on a healthy GPU, so a format
  // that looks slow is burned once more on every GPU; SLOW stands only if it repeats.
  std::vector<FormatSpec> suspicious;
  for (const FormatSpec& f : formats) {
    for (const FormatResult& r : results) {
      if (r.format == f.name && r.verdict == Verdict::slow) {
        suspicious.push_back(f);
        break;
      }
    }
  }
  if (!suspicious.empty() && !g_stop) {
    if (!quiet) {
      std::printf("\nConfirming %zu slow-looking format(s) with a second run:", suspicious.size());
      for (const auto& f : suspicious) std::printf(" %s", f.name.c_str());
      std::printf("\n");
    }
    confirm_slow(results, burn(suspicious, "#confirm"));
  }
  monitor.finish();
  apply_memory_errors(results, devices, monitor);
  if (!quiet) std::puts("done\n");

  RunSummary summary{devices, results, slice, &monitor, libraries};
  if (opt.json) print_json(summary);
  else print_text(summary);
  return exit_code(summary);
}

}  // namespace
}  // namespace brrr

int main(int argc, char** argv) {
  try {
    return brrr::run(brrr::parse_args(argc, argv));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "brrr: %s\n", e.what());
    return 3;
  }
}
