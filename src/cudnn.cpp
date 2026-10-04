// cuDNN matmul for block-scaled formats cuBLASLt has no kernels for (MXFP4 on Blackwell).
// libcudnn is opened at runtime, so brrr runs without it; those formats are then unsupported.

#include <dlfcn.h>

#include <stdexcept>
#include <string>

#include "backend.hpp"

#ifdef BRRR_CUDNN

#define NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
#include <cudnn_frontend.h>

#include "check.hpp"
#include "kernels.cuh"

namespace cudnn_frontend {
void* cudnn_dlhandle = nullptr;
}

namespace brrr {
namespace {

namespace fe = cudnn_frontend;

constexpr const char* kSoname = "libcudnn.so.9";

// Opens libcudnn once per process; returns why not when it cannot.
const std::string& load_error() {
  static const std::string error = [] {
    fe::cudnn_dlhandle = dlopen(kSoname, RTLD_NOW | RTLD_GLOBAL);
    return fe::cudnn_dlhandle ? std::string() : std::string("cuDNN: ") + kSoname + " not found";
  }();
  return error;
}

fe::DataType_t fe_type(DType t) {
  switch (t) {
    case DType::fp32: return fe::DataType_t::FLOAT;
    case DType::fp16: return fe::DataType_t::HALF;
    case DType::bf16: return fe::DataType_t::BFLOAT16;
    case DType::e4m3: return fe::DataType_t::FP8_E4M3;
    case DType::e2m1: return fe::DataType_t::FP4_E2M1;
    default: throw std::runtime_error("cuDNN backend: type " + to_string(t) + " not handled");
  }
}

size_t storage_bytes(DType t, size_t count) { return (count * bits(t) + 7) / 8; }
long ceil_div(long v, long by) { return (v + by - 1) / by; }
long round_up(long v, long to) { return ceil_div(v, to) * to; }

// Only the block-scaled formats Blackwell runs natively: a 1 x N block along K, with
// E8M0 (MX) or E4M3 (NVFP4) scales. cuDNN's generic dequantize path is too slow to burn.
bool native(const FormatSpec& f) {
  const bool inputs = (f.scale == BlockScale::ue8m0 && bits(f.type) <= 8) ||
                      (f.scale == BlockScale::ue4m3 && f.type == DType::e2m1);
  return inputs && bits(f.out) > 8;
}

void check(fe::error_t status) {
  if (status.is_good()) return;
  std::string why = status.get_message();
  why = why.substr(0, why.find('\n'));
  throw std::runtime_error("cuDNN: " + (why.empty() ? std::string("no engine for this format on this GPU") : why));
}

class Cudnn final : public Backend {
 public:
  ~Cudnn() override { release(); }

  const char* name() const override { return "cuDNN"; }

  Support probe(const FormatSpec& spec) override {
    if (!native(spec)) return {false, "cuDNN: not a native block-scaled format"};
    if (!load_error().empty()) return {false, load_error()};
    try {
      describe(spec);
      return {true, ""};
    } catch (const std::exception& e) {
      const std::string why = e.what();
      return {false, why.rfind("cuDNN: ", 0) == 0 ? why : "cuDNN: " + why};
    }
  }

  void setup(const FormatSpec& spec, uint64_t seed) override {
    describe(spec);
    const size_t a_scales = size_t(sm_ * sk_), b_scales = size_t(sk_ * sn_);
    for (auto [p, n] : {std::pair{&a_, storage_bytes(spec.type, size_t(m_) * k_)},
                        std::pair{&b_, storage_bytes(spec.type, size_t(k_) * n_)},
                        std::pair{&a_scale_, a_scales}, std::pair{&b_scale_, b_scales}}) {
      check_cuda(cudaMalloc(p, n), "alloc");
    }
    int64_t workspace = 0;
    check(graph_.get_workspace_size(workspace));
    if (workspace > 0) check_cuda(cudaMalloc(&workspace_, size_t(workspace)), "alloc workspace");
    fill_random(a_, spec.type, size_t(m_) * k_, seed, nullptr);
    fill_random(b_, spec.type, size_t(k_) * n_, seed ^ 0x5bd1e995, nullptr);
    fill_scales(a_scale_, spec.scale, a_scales, seed ^ 0xa11ce, nullptr);
    fill_scales(b_scale_, spec.scale, b_scales, seed ^ 0xb0b, nullptr);
    check_cuda(cudaDeviceSynchronize(), "setup");
    pack_ = {{a_t_, a_}, {b_t_, b_}, {sa_t_, a_scale_}, {sb_t_, b_scale_}, {d_t_, nullptr}};
  }

  void run(cudaStream_t stream, void* out) override {
    if (fe::detail::set_stream(handle_, stream) != CUDNN_STATUS_SUCCESS) throw std::runtime_error("cudnnSetStream");
    pack_[d_t_] = out;
    check(graph_.execute(handle_, pack_, workspace_));
  }

  double flops() const override { return 2.0 * m_ * n_ * k_; }
  size_t out_bytes() const override { return storage_bytes(out_, size_t(m_) * n_); }

 private:
  // Builds the dequantize + matmul graph and lets cuDNN pick its plan; throws with the
  // reason when nothing can run it here.
  void describe(const FormatSpec& f) {
    release();
    out_ = f.out;
    // Scales are stored in cuDNN's 128 x 4 tiles, padded like cuBLASLt's.
    const long block = block_size(f.scale);
    sm_ = round_up(m_, 128);
    sn_ = round_up(n_, 128);
    sk_ = round_up(ceil_div(k_, block), 4);
    if (fe::detail::create_handle(&handle_) != CUDNN_STATUS_SUCCESS) throw std::runtime_error("cudnnCreate failed");

    graph_ = fe::graph::Graph();
    graph_.set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
    using T = fe::graph::Tensor_attributes;
    const fe::DataType_t scale = f.scale == BlockScale::ue8m0 ? fe::DataType_t::FP8_E8M0 : fe::DataType_t::FP8_E4M3;
    a_t_ = graph_.tensor(T().set_name("A").set_data_type(fe_type(f.type)).set_dim({1, m_, k_}).set_stride({m_ * k_, k_, 1}));
    b_t_ = graph_.tensor(T().set_name("B").set_data_type(fe_type(f.type)).set_dim({1, k_, n_}).set_stride({k_ * n_, 1, k_}));
    sa_t_ = graph_.tensor(T().set_name("A scales").set_data_type(scale).set_dim({1, sm_, sk_})
                              .set_stride({sm_ * sk_, sk_, 1}).set_reordering_type(fe::TensorReordering_t::F8_128x4));
    sb_t_ = graph_.tensor(T().set_name("B scales").set_data_type(scale).set_dim({1, sk_, sn_})
                              .set_stride({sk_ * sn_, 1, sk_}).set_reordering_type(fe::TensorReordering_t::F8_128x4));
    auto a = graph_.block_scale_dequantize(a_t_, sa_t_, fe::graph::Block_scale_dequantize_attributes().set_block_size({1, int32_t(block)}));
    auto b = graph_.block_scale_dequantize(b_t_, sb_t_, fe::graph::Block_scale_dequantize_attributes().set_block_size({int32_t(block), 1}));
    d_t_ = graph_.matmul(a, b, fe::graph::Matmul_attributes().set_compute_data_type(fe::DataType_t::FLOAT));
    d_t_->set_data_type(fe_type(f.out)).set_is_virtual(false);

    check(graph_.validate());
    check(graph_.build_operation_graph(handle_));
    check(graph_.create_execution_plans({fe::HeurMode_t::A}));
    check(graph_.check_support());
    check(graph_.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));
  }

  void release() {
    for (void* p : {a_, b_, a_scale_, b_scale_, workspace_}) {
      if (p) cudaFree(p);
    }
    a_ = b_ = a_scale_ = b_scale_ = workspace_ = nullptr;
    if (handle_) fe::detail::destroy_handle(handle_);
    handle_ = nullptr;
  }

  static constexpr long m_ = kSize, n_ = kSize, k_ = kSize;
  long sm_ = 0, sn_ = 0, sk_ = 0;
  DType out_ = DType::bf16;
  cudnnHandle_t handle_ = nullptr;
  fe::graph::Graph graph_;
  std::shared_ptr<fe::graph::Tensor_attributes> a_t_, b_t_, sa_t_, sb_t_, d_t_;
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> pack_;
  void *a_ = nullptr, *b_ = nullptr, *a_scale_ = nullptr, *b_scale_ = nullptr, *workspace_ = nullptr;
};

}  // namespace

std::unique_ptr<Backend> make_cudnn() { return std::make_unique<Cudnn>(); }

std::optional<Library> cudnn_library() {
  if (!load_error().empty()) return std::nullopt;
  const size_t v = fe::detail::get_backend_version();
  Library lib{"cuDNN", std::to_string(v / 10000) + "." + std::to_string(v / 100 % 100) + "." +
                           std::to_string(v % 100), kSoname};
  Dl_info info;
  if (void* sym = dlsym(fe::cudnn_dlhandle, "cudnnGetVersion"); sym && dladdr(sym, &info) && info.dli_fname) {
    lib.path = info.dli_fname;
  }
  return lib;
}

}  // namespace brrr

#else  // built without cuDNN headers

namespace brrr {

namespace {
class NoCudnn final : public Backend {
 public:
  const char* name() const override { return "cuDNN"; }
  Support probe(const FormatSpec&) override { return {false, "cuDNN: not built in"}; }
  void setup(const FormatSpec&, uint64_t) override {}
  void run(cudaStream_t, void*) override {}
  double flops() const override { return 0; }
  size_t out_bytes() const override { return 0; }
};
}  // namespace

std::unique_ptr<Backend> make_cudnn() { return std::make_unique<NoCudnn>(); }
std::optional<Library> cudnn_library() { return std::nullopt; }

}  // namespace brrr

#endif
