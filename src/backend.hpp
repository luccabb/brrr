#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "format.hpp"

namespace brrr {

constexpr long kSize = 8192;

struct Device {
  int index = 0;
  int arch = 0;  // compute capability x 10
  std::string name;
  size_t memory_bytes = 0;
  std::string pci_bus_id;  // matches CUDA devices to NVML devices
};

struct Support {
  bool ok = true;
  std::string reason;  // set when !ok
};

// One way of running a format's multiply. Implementations must produce identical bits
// on every run() for the same setup(); the worker checks that before trusting them.
class Backend {
 public:
  virtual ~Backend() = default;
  virtual const char* name() const = 0;
  virtual Support probe(const FormatSpec& spec) = 0;
  virtual void setup(const FormatSpec& spec, uint64_t seed) = 0;
  virtual void run(cudaStream_t stream, void* out) = 0;
  virtual double flops() const = 0;
  virtual size_t out_bytes() const = 0;
};

std::unique_ptr<Backend> make_cublaslt();
std::unique_ptr<Backend> make_cudnn();

// cuBLASLt, or cuDNN for the block-scaled formats cuBLASLt has no kernels for. Returns
// nullptr, with the reason in `support`, when neither can run `spec` on the current GPU.
std::unique_ptr<Backend> pick_backend(const FormatSpec& spec, Support& support);

struct Library {
  std::string name;     // e.g. cuBLASLt
  std::string version;  // e.g. 13.0.2
  std::string path;     // the file the loader picked
};

// The cuBLASLt the loader picked, so every result says what produced it.
Library cublaslt_library();
std::optional<Library> cudnn_library();  // when libcudnn can be loaded

}  // namespace brrr
