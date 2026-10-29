// Runtime dispatch between the scalar and AVX2 kernels.
#include "fastframe/preprocess.hpp"

#include <cstdlib>
#include <cstring>

namespace ff {

namespace {
bool detect_avx2() noexcept {
#if defined(__x86_64__) || defined(__i386__)
  // FF_FORCE_SCALAR=1 lets benchmarks and tests exercise the scalar path on AVX2 machines.
  if (const char* e = std::getenv("FF_FORCE_SCALAR"); e != nullptr && e[0] == '1') return false;
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") != 0;
#else
  return false;
#endif
}
const bool kHasAvx2 = detect_avx2();
}  // namespace

bool cpu_has_avx2() noexcept { return kHasAvx2; }

std::string_view preprocess_backend() noexcept {
#if defined(FF_HAVE_AVX2_TU)
  return kHasAvx2 ? "avx2" : "scalar";
#else
  return "scalar";
#endif
}

void normalize_chw(std::span<const std::uint8_t> bgr, std::span<float> chw, int w, int h) noexcept {
#if defined(FF_HAVE_AVX2_TU)
  if (kHasAvx2) { normalize_chw_avx2(bgr, chw, w, h); return; }
#endif
  normalize_chw_scalar(bgr, chw, w, h);
}

void letterbox_resize(std::span<const std::uint8_t> src, int src_w, int src_h,
                      std::span<std::uint8_t> dst, int model, const Letterbox& lb) noexcept {
#if defined(FF_HAVE_AVX2_TU)
  if (kHasAvx2) { letterbox_resize_avx2(src, src_w, src_h, dst, model, lb); return; }
#endif
  letterbox_resize_scalar(src, src_w, src_h, dst, model, lb);
}

void preprocess_frame_twopass(std::span<const std::uint8_t> src, int src_w, int src_h, int model,
                              std::span<std::uint8_t> scratch, std::span<float> out, Letterbox& lb) noexcept {
  lb = compute_letterbox(src_w, src_h, model);
  letterbox_resize_scalar(src, src_w, src_h, scratch, model, lb);
  normalize_chw_scalar(scratch, out, model, model);
}

void preprocess_frame(std::span<const std::uint8_t> src, int src_w, int src_h, int model,
                      std::span<std::uint8_t> scratch, std::span<float> out, Letterbox& lb) noexcept {
#if defined(FF_HAVE_AVX2_TU)
  if (kHasAvx2) {
    lb = compute_letterbox(src_w, src_h, model);
    preprocess_fused_avx2(src, src_w, src_h, model, out, lb);
    return;
  }
#endif
  preprocess_frame_twopass(src, src_w, src_h, model, scratch, out, lb);
}

}  // namespace ff
