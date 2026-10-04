#include "worker.hpp"

#include <cuda_runtime.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>

#include "backend.hpp"
#include "check.hpp"
#include "kernels.cuh"

namespace brrr {
namespace {

using Clock = std::chrono::steady_clock;

constexpr double kWarmupFraction = 0.10;
constexpr double kReportEverySeconds = 1.0;
constexpr int kInFlight = 2;  // iterations queued ahead of the GPU, so slices end on time

void send(int fd, Record r) {
  // Records are far below PIPE_BUF, so each write is atomic.
  [[maybe_unused]] ssize_t n = ::write(fd, &r, sizeof r);
}

void set_reason(Record& r, const std::string& why) {
  std::strncpy(r.reason, why.c_str(), sizeof r.reason - 1);
}

Device describe_device(int gpu) {
  cudaDeviceProp prop{};
  check_cuda(cudaGetDeviceProperties(&prop, gpu), "device properties");
  return {gpu, prop.major * 10 + prop.minor, prop.name, prop.totalGlobalMem};
}

// Device buffers for one format's run: the reference output, a ring of outputs filling
// most of the GPU's memory (each multiply writes the next one, so every part of memory is
// written and checked, as in gpu-burn), and the running mismatch counter.
struct Buffers {
  void* reference = nullptr;
  char* ring = nullptr;
  size_t out_bytes = 0;
  size_t slots = 1;
  unsigned long long* mismatches = nullptr;

  Buffers(size_t out, const WorkerPlan& plan) : out_bytes(out) {
    check_cuda(cudaMalloc(&reference, out_bytes), "alloc reference");
    check_cuda(cudaMalloc(&mismatches, sizeof *mismatches), "alloc counter");
    check_cuda(cudaMemset(mismatches, 0, sizeof *mismatches), "zero counter");
    size_t free = 0, total = 0;
    check_cuda(cudaMemGetInfo(&free, &total), "memory info");
    const size_t target = plan.memory_bytes ? plan.memory_bytes : size_t(free * plan.memory_fraction);
    slots = std::max<size_t>(1, target > out_bytes ? (target - out_bytes) / out_bytes : 1);
    while (cudaMalloc(reinterpret_cast<void**>(&ring), slots * out_bytes) != cudaSuccess) {
      cudaGetLastError();
      if (slots == 1) throw std::runtime_error("cannot allocate one output");
      slots = slots * 9 / 10;
    }
  }
  ~Buffers() {
    cudaFree(reference);
    cudaFree(ring);
    cudaFree(mismatches);
  }
  void* slot(long iteration) const { return ring + size_t(iteration) % slots * out_bytes; }
  Buffers(const Buffers&) = delete;
  Buffers& operator=(const Buffers&) = delete;

  unsigned long long read_mismatches() const {
    unsigned long long n = 0;
    check_cuda(cudaMemcpy(&n, mismatches, sizeof n, cudaMemcpyDeviceToHost), "read counter");
    return n;
  }
};

void burn_format(const WorkerPlan& plan, int index, const Device& device, int fd) {
  const FormatSpec& spec = plan.formats[index];
  Record rec;
  rec.gpu = plan.gpu;
  rec.format = index;

  Support support;
  std::unique_ptr<Backend> backend = pick_backend(spec, support);
  if (!backend) {
    rec.kind = Record::Kind::unsupported;
    set_reason(rec, support.reason);
    return send(fd, rec);
  }
  std::strncpy(rec.backend, backend->name(), sizeof rec.backend - 1);

  size_t free = 0, total = 0;
  check_cuda(cudaMemGetInfo(&free, &total), "memory info");
  rec.free_bytes = int64_t(free);
  rec.total_bytes = int64_t(total);
  try {
    backend->setup(spec, plan.seed);
  } catch (const Refused& e) {
    rec.kind = Record::Kind::unsupported;
    set_reason(rec, e.what());
    return send(fd, rec);
  }
  const size_t out_bytes = backend->out_bytes();
  Buffers buf(out_bytes, plan);
  rec.memory_bytes = int64_t((buf.slots + 1) * out_bytes);
  cudaStream_t stream = nullptr;
  check_cuda(cudaStreamCreate(&stream), "stream");

  // Repeatability gate: a format whose result changes between two runs cannot be checked.
  backend->run(stream, buf.reference);
  backend->run(stream, buf.slot(0));
  count_mismatches(buf.reference, buf.slot(0), out_bytes, buf.mismatches, stream);
  check_cuda(cudaStreamSynchronize(stream), "repeatability gate");
  if (buf.read_mismatches() != 0) {
    rec.kind = Record::Kind::unsupported;
    set_reason(rec, "not repeatable: two runs on the same input differ");
    cudaStreamDestroy(stream);
    return send(fd, rec);
  }

  // Per in-flight slot: multiply start/end (timed, so TFLOPS excludes the compare) and
  // a marker after the compare that bounds how far the host runs ahead.
  cudaEvent_t begin[kInFlight], finish[kInFlight], queued[kInFlight];
  bool timed[kInFlight] = {};
  for (int i = 0; i < kInFlight; ++i) {
    check_cuda(cudaEventCreate(&begin[i]), "event");
    check_cuda(cudaEventCreate(&finish[i]), "event");
    check_cuda(cudaEventCreateWithFlags(&queued[i], cudaEventDisableTiming), "event");
  }
  double multiply_ms = 0;
  long measured_iterations = 0;
  auto collect = [&](int slot) {
    if (!timed[slot]) return;
    float ms = 0;
    check_cuda(cudaEventElapsedTime(&ms, begin[slot], finish[slot]), "elapsed");
    multiply_ms += ms;
    ++measured_iterations;
    timed[slot] = false;
  };

  const auto t0 = Clock::now();
  const auto warmup_end = t0 + std::chrono::duration<double>(plan.seconds_per_format * kWarmupFraction);
  const auto end = t0 + std::chrono::duration<double>(plan.seconds_per_format);
  auto next_report = t0 + std::chrono::duration<double>(kReportEverySeconds);

  while (Clock::now() < end) {
    const int slot = static_cast<int>(rec.iterations % kInFlight);
    if (rec.iterations >= kInFlight) {
      check_cuda(cudaEventSynchronize(queued[slot]), "in-flight wait");
      collect(slot);
    }
    const bool measure = Clock::now() >= warmup_end;
    void* out = buf.slot(rec.iterations);
    if (measure) check_cuda(cudaEventRecord(begin[slot], stream), "begin event");
    backend->run(stream, out);
    if (measure) check_cuda(cudaEventRecord(finish[slot], stream), "finish event");
    timed[slot] = measure;
    if (rec.iterations == plan.inject_fault_after) corrupt_one_byte(out, out_bytes / 2, stream);
    count_mismatches(buf.reference, out, out_bytes, buf.mismatches, stream);
    check_cuda(cudaEventRecord(queued[slot], stream), "queue event");
    ++rec.iterations;

    if (Clock::now() >= next_report) {
      check_cuda(cudaStreamSynchronize(stream), "progress");
      rec.errors = static_cast<int64_t>(buf.read_mismatches());
      rec.tflops = multiply_ms > 0 ? backend->flops() * measured_iterations / (multiply_ms * 1e-3) / 1e12 : 0;
      rec.slice_done = float(std::chrono::duration<double>(Clock::now() - t0).count() / plan.seconds_per_format);
      send(fd, rec);
      next_report += std::chrono::duration<double>(kReportEverySeconds);
    }
  }

  check_cuda(cudaStreamSynchronize(stream), "final sync");
  for (int i = 0; i < kInFlight; ++i) collect(i);
  const double ms = multiply_ms;
  rec.kind = Record::Kind::done;
  rec.errors = static_cast<int64_t>(buf.read_mismatches());
  rec.tflops = ms > 0 ? backend->flops() * measured_iterations / (ms * 1e-3) / 1e12 : 0;
  send(fd, rec);

  for (int i = 0; i < kInFlight; ++i) {
    for (cudaEvent_t e : {begin[i], finish[i], queued[i]}) cudaEventDestroy(e);
  }
  cudaStreamDestroy(stream);
}

}  // namespace

int run_worker(const WorkerPlan& plan, int fd) {
  int index = 0;
  try {
    check_cuda(cudaSetDevice(plan.gpu), "set device");
    const Device device = describe_device(plan.gpu);
    for (; index < static_cast<int>(plan.formats.size()); ++index) {
      burn_format(plan, index, device, fd);
    }
    return 0;
  } catch (const std::exception& e) {
    Record rec;
    rec.kind = Record::Kind::failed;
    rec.gpu = plan.gpu;
    rec.format = index;
    set_reason(rec, e.what());
    send(fd, rec);
    return 1;
  }
}

}  // namespace brrr
