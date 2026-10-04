#include "backend.hpp"

namespace brrr {

std::unique_ptr<Backend> pick_backend(const FormatSpec& spec, Support& support) {
  std::unique_ptr<Backend> lt = make_cublaslt();
  support = lt->probe(spec);
  if (support.ok) return lt;
  std::unique_ptr<Backend> dnn = make_cudnn();
  const Support s = dnn->probe(spec);
  if (s.ok) {
    support = s;
    return dnn;
  }
  if (s.reason.rfind("cuDNN: ", 0) == 0 && s.reason.find("not a native") == std::string::npos) {
    support.reason += "; " + s.reason;
  }
  return nullptr;
}

}  // namespace brrr
