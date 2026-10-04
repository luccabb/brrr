#pragma once

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace brrr {

inline void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

// cuBLASLt refusing a call: the library cannot run this format, the GPU is not at fault.
struct Refused : std::runtime_error {
  using std::runtime_error::runtime_error;
};

inline void check_cublas(cublasStatus_t status, const char* what) {
  const std::string message = std::string(what) + ": " + cublasLtGetStatusString(status);
  switch (status) {
    case CUBLAS_STATUS_SUCCESS: return;
    case CUBLAS_STATUS_INVALID_VALUE:
    case CUBLAS_STATUS_NOT_SUPPORTED:
    case CUBLAS_STATUS_ARCH_MISMATCH: throw Refused(message);
    default: throw std::runtime_error(message);
  }
}

}  // namespace brrr
