// cuBLASLt matmul for every format it can express. Split-K reductions done in place with
// atomics are excluded, so results repeat bit for bit (the worker's gate double-checks).

#include <cublasLt.h>
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend.hpp"
#include "check.hpp"
#include "kernels.cuh"

namespace brrr {
namespace {

constexpr size_t kWorkspaceBytes = 64ull << 20;
constexpr int kCandidates = 8;  // repeatable algorithms tried during setup
constexpr int kTuneRepeats = 5;

// cuBLASLt explains rejected configurations only through its logger; keep the last error so
// "unsupported" reasons say why. CUBLASLT_LOG_LEVEL, cuBLASLt's own switch, prints the log.
thread_local std::string g_last_cublaslt_error;
const char* const kLogLevel = std::getenv("CUBLASLT_LOG_LEVEL");

void capture_log(int level, const char* function, const char* message) {
  if (level == 1) g_last_cublaslt_error = std::string(function) + ": " + message;
  if (kLogLevel) std::fprintf(stderr, "[cublasLt %d] %s: %s\n", level, function, message);
}

bool is_fp8(DType t) { return t == DType::e4m3; }
bool is_fp4(DType t) { return t == DType::e2m1; }

cudaDataType_t cuda_type(DType t) {
  switch (t) {
    case DType::fp64: return CUDA_R_64F;
    case DType::fp32: case DType::tf32: return CUDA_R_32F;
    case DType::fp16: return CUDA_R_16F;
    case DType::bf16: return CUDA_R_16BF;
    case DType::e4m3: return CUDA_R_8F_E4M3;
    case DType::e2m1: return CUDA_R_4F_E2M1;
  }
  throw std::runtime_error("type " + to_string(t) + " has no cuBLASLt type");
}

size_t storage_bytes(DType t, size_t count) { return count * bits(t) / 8; }

long round_up(long v, long to) { return (v + to - 1) / to * to; }
long ceil_div(long v, long by) { return (v + by - 1) / by; }

// How one operand is scaled, in cuBLASLt's terms: one FP32 scale for the tensor, or a
// block scale per 32 / 16 values along K. `rows` is the operand's outer dimension.
struct ScalePlan {
  bool present = false;
  cublasLtMatmulMatrixScale_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
  BlockScale block = BlockScale::none;
  size_t count = 0;
};

const ScalePlan kTensorScale = {true, CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F, BlockScale::none, 1};

ScalePlan plan_scale(const FormatSpec& f, long rows, long k) {
  if (f.scale == BlockScale::none) {
    if (is_fp4(f.type)) throw std::runtime_error("e2m1 needs block scales");
    return is_fp8(f.type) ? kTensorScale : ScalePlan{};
  }
  // Buffers are padded to the tiles cuBLASLt reads (128 rows x 4 blocks).
  const long n = block_size(f.scale);
  return {true, f.scale == BlockScale::ue8m0 ? CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0
                                             : CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3,
          f.scale, size_t(round_up(rows, 128) * round_up(ceil_div(k, n), 4))};
}

size_t scale_bytes(const ScalePlan& p) {
  return p.block == BlockScale::none ? p.count * sizeof(float) : p.count;
}

cublasComputeType_t compute_type(const FormatSpec& f) {
  if (f.type == DType::fp64) return CUBLAS_COMPUTE_64F;
  if (f.type == DType::tf32) return CUBLAS_COMPUTE_32F_FAST_TF32;
  if (f.type == DType::fp32) return CUBLAS_COMPUTE_32F_PEDANTIC;  // CUDA cores, like gpu-burn
  return CUBLAS_COMPUTE_32F;
}

class CublasLt final : public Backend {
 public:
  ~CublasLt() override { release(); }

  const char* name() const override { return "cuBLASLt"; }

  Support probe(const FormatSpec& spec) override {
    try {
      describe(spec);
      return {true, ""};
    } catch (const std::exception& e) {
      return {false, e.what()};
    }
  }

  void setup(const FormatSpec& spec, uint64_t seed) override {
    describe(spec);
    check_cuda(cudaMalloc(&a_, storage_bytes(spec.type, size_t(m_) * k_)), "alloc A");
    check_cuda(cudaMalloc(&b_, storage_bytes(spec.type, size_t(k_) * n_)), "alloc B");
    check_cuda(cudaMalloc(&workspace_, kWorkspaceBytes), "alloc workspace");
    fill_random(a_, spec.type, size_t(m_) * k_, seed, nullptr);
    fill_random(b_, spec.type, size_t(k_) * n_, seed ^ 0x5bd1e995, nullptr);
    fill(a_plan_, a_scale_, seed ^ 0xa11ce);
    fill(b_plan_, b_scale_, seed ^ 0xb0b);
    if (d_scale_) {
      // Inputs are uniform in [-1, 1), so outputs have a standard deviation of about
      // sqrt(K) / 3. Scale them to about 16: well inside FP8's range (e4m3 tops out at
      // 448), even with input block scales of up to 2, and still using its precision.
      const float d_scale = 48.0f / std::sqrt(float(k_));
      check_cuda(cudaMemcpy(d_scale_, &d_scale, sizeof d_scale, cudaMemcpyHostToDevice), "output scale");
    }
    check_cuda(cudaDeviceSynchronize(), "setup");
    tune();
  }

  void run(cudaStream_t stream, void* out) override {
    check_cublas(cublasLtMatmul(handle_, desc_, alpha(), a_, a_layout_, b_, b_layout_, beta(),
                                d_scale_ ? nullptr : out, c_layout_, out, d_layout_, &algo_, workspace_,
                                kWorkspaceBytes, stream),
                 "cublasLtMatmul");
  }

  double flops() const override { return 2.0 * m_ * n_ * k_; }
  size_t out_bytes() const override { return storage_bytes(out_, size_t(m_) * n_); }

 private:
  // Builds descriptors and keeps every repeatable algorithm cuBLASLt suggests; throws
  // with the reason when the format cannot run here.
  void describe(const FormatSpec& f) {
    release();
    out_ = f.out;
    fp64_ = f.type == DType::fp64;
    a_plan_ = plan_scale(f, m_, k_);
    b_plan_ = plan_scale(f, n_, k_);

    static FILE* const kNoLogFile = std::fopen("/dev/null", "w");
    check_cublas(cublasLtLoggerSetFile(kNoLogFile), "logger file");
    check_cublas(cublasLtLoggerSetCallback(capture_log), "logger");
    check_cublas(cublasLtLoggerSetLevel(kLogLevel ? std::max(1, std::atoi(kLogLevel)) : 1), "log level");
    g_last_cublaslt_error.clear();
    check_cublas(cublasLtCreate(&handle_), "create handle");
    const cudaDataType_t scale_type = fp64_ ? CUDA_R_64F : CUDA_R_32F;
    check_cublas(cublasLtMatmulDescCreate(&desc_, compute_type(f), scale_type), "matmul desc");
    // A is stored K x M and transposed ("TN"), the layout 8- and 4-bit inputs require.
    const cublasOperation_t transpose = CUBLAS_OP_T;
    set(CUBLASLT_MATMUL_DESC_TRANSA, transpose);
    // Block-scale modes need their scale pointers before cuBLASLt will suggest algorithms.
    a_scale_ = attach_scales(a_plan_, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER);
    b_scale_ = attach_scales(b_plan_, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER);
    if (is_fp8(f.out)) {
      d_scale_ = attach_scales(kTensorScale, CUBLASLT_MATMUL_DESC_D_SCALE_MODE, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER);
    }
    check_cublas(cublasLtMatrixLayoutCreate(&a_layout_, cuda_type(f.type), k_, m_, k_), "A layout");
    check_cublas(cublasLtMatrixLayoutCreate(&b_layout_, cuda_type(f.type), k_, n_, k_), "B layout");
    check_cublas(cublasLtMatrixLayoutCreate(&d_layout_, cuda_type(f.out), m_, n_, m_), "D layout");
    // cuBLASLt has no FP8 C; with beta = 0 C is never read, so FP8 outputs pass none.
    const DType c_type = d_scale_ ? DType::bf16 : f.out;
    check_cublas(cublasLtMatrixLayoutCreate(&c_layout_, cuda_type(c_type), m_, n_, m_), "C layout");

    cublasLtMatmulPreference_t pref = nullptr;
    check_cublas(cublasLtMatmulPreferenceCreate(&pref), "preference");
    const uint64_t workspace = kWorkspaceBytes;
    const uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_MASK & ~CUBLASLT_REDUCTION_SCHEME_INPLACE;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                         &workspace, sizeof workspace);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                                         &reduction, sizeof reduction);
    cublasLtMatmulHeuristicResult_t results[kCandidates] = {};
    int found = 0;
    const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
        handle_, desc_, a_layout_, b_layout_, c_layout_, d_layout_, pref, kCandidates, results, &found);
    cublasLtMatmulPreferenceDestroy(pref);
    if (status != CUBLAS_STATUS_SUCCESS || found == 0) {
      throw std::runtime_error(g_last_cublaslt_error.empty()
                                   ? "cuBLASLt offers no algorithm for this combination on this GPU and cuBLAS version"
                                   : g_last_cublaslt_error);
    }
    candidates_.assign(results, results + found);
    algo_ = candidates_.front().algo;
  }

  template <typename T>
  void set(cublasLtMatmulDescAttributes_t attribute, const T& value) {
    check_cublas(cublasLtMatmulDescSetAttribute(desc_, attribute, &value, sizeof value), "set attribute");
  }

  // Allocates one operand's scale buffer and points the descriptor at it.
  void* attach_scales(const ScalePlan& plan, cublasLtMatmulDescAttributes_t mode,
                      cublasLtMatmulDescAttributes_t pointer) {
    if (!plan.present) return nullptr;
    void* scales = nullptr;
    check_cuda(cudaMalloc(&scales, scale_bytes(plan)), "alloc scales");
    set(mode, int32_t(plan.mode));
    set(pointer, scales);
    return scales;
  }

  static void fill(const ScalePlan& plan, void* scales, uint64_t seed) {
    if (!plan.present) return;
    if (plan.mode == CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F) {
      const float one = 1.0f;
      check_cuda(cudaMemcpy(scales, &one, sizeof one, cudaMemcpyHostToDevice), "scalar scale");
    } else {
      fill_scales(scales, plan.block, plan.count, seed, nullptr);
    }
  }

  // Keeps the fastest candidate. None reduce in place, so all should repeat exactly.
  void tune() {
    void* out = nullptr;
    check_cuda(cudaMalloc(&out, out_bytes()), "alloc tune output");
    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "event");
    check_cuda(cudaEventCreate(&stop), "event");
    float best_ms = 0;
    cublasLtMatmulAlgo_t best = candidates_.front().algo;
    for (const auto& candidate : candidates_) {
      algo_ = candidate.algo;
      run(nullptr, out);  // warm-up
      check_cuda(cudaEventRecord(start), "event");
      for (int i = 0; i < kTuneRepeats; ++i) run(nullptr, out);
      check_cuda(cudaEventRecord(stop), "event");
      check_cuda(cudaEventSynchronize(stop), "tune");
      float ms = 0;
      check_cuda(cudaEventElapsedTime(&ms, start, stop), "elapsed");
      if (best_ms == 0 || ms < best_ms) {
        best_ms = ms;
        best = candidate.algo;
      }
    }
    algo_ = best;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(out);
  }

  const void* alpha() const { return fp64_ ? static_cast<const void*>(&one_d_) : &one_f_; }
  const void* beta() const { return fp64_ ? static_cast<const void*>(&zero_d_) : &zero_f_; }

  void release() {
    for (void* p : {a_, b_, workspace_, a_scale_, b_scale_, d_scale_}) {
      if (p) cudaFree(p);
    }
    a_ = b_ = workspace_ = a_scale_ = b_scale_ = d_scale_ = nullptr;
    if (a_layout_) cublasLtMatrixLayoutDestroy(a_layout_);
    if (b_layout_) cublasLtMatrixLayoutDestroy(b_layout_);
    if (c_layout_) cublasLtMatrixLayoutDestroy(c_layout_);
    if (d_layout_) cublasLtMatrixLayoutDestroy(d_layout_);
    if (desc_) cublasLtMatmulDescDestroy(desc_);
    if (handle_) cublasLtDestroy(handle_);
    a_layout_ = b_layout_ = c_layout_ = d_layout_ = nullptr;
    desc_ = nullptr;
    handle_ = nullptr;
  }

  static constexpr long m_ = kSize, n_ = kSize, k_ = kSize;
  DType out_ = DType::bf16;
  bool fp64_ = false;
  ScalePlan a_plan_, b_plan_;
  float one_f_ = 1.0f, zero_f_ = 0.0f;
  double one_d_ = 1.0, zero_d_ = 0.0;
  cublasLtHandle_t handle_ = nullptr;
  cublasLtMatmulDesc_t desc_ = nullptr;
  cublasLtMatrixLayout_t a_layout_ = nullptr, b_layout_ = nullptr, c_layout_ = nullptr, d_layout_ = nullptr;
  cublasLtMatmulAlgo_t algo_{};
  std::vector<cublasLtMatmulHeuristicResult_t> candidates_;
  void *a_ = nullptr, *b_ = nullptr, *workspace_ = nullptr, *a_scale_ = nullptr, *b_scale_ = nullptr,
       *d_scale_ = nullptr;
};

}  // namespace

std::unique_ptr<Backend> make_cublaslt() { return std::make_unique<CublasLt>(); }

Library cublaslt_library() {
  const size_t v = cublasLtGetVersion();
  Library lib{"cuBLASLt", std::to_string(v / 10000) + "." + std::to_string(v / 100 % 100) + "." +
                              std::to_string(v % 100), "unknown"};
  Dl_info info;
  if (dladdr(reinterpret_cast<void*>(&cublasLtGetVersion), &info) && info.dli_fname) lib.path = info.dli_fname;
  return lib;
}

}  // namespace brrr
