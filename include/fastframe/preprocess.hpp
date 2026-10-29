#pragma once
// Preprocessing: letterbox-resize BGR uint8 HWC -> RGB float32 CHW in [0,1].
// Two implementations with identical output: a scalar reference and an AVX2
// kernel, chosen at runtime by CPU feature detection.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace ff {

/// Letterbox geometry for mapping detections back to source pixels.
struct Letterbox {
  float scale;    ///< source * scale = resized
  int pad_x;      ///< left padding in model pixels
  int pad_y;      ///< top padding
  int new_w;      ///< resized width (before padding)
  int new_h;
};

[[nodiscard]] Letterbox compute_letterbox(int src_w, int src_h, int model) noexcept;

/// Nearest-neighbour letterbox resize BGR->BGR (uint8 HWC), fills padding with 114.
/// Output is model x model x 3. Same algorithm in scalar and AVX2 so results are bit-identical.
void letterbox_resize_scalar(std::span<const std::uint8_t> src, int src_w, int src_h,
                             std::span<std::uint8_t> dst, int model, const Letterbox& lb) noexcept;

/// BGR uint8 HWC (w*h*3) -> RGB float32 CHW (3*w*h), divided by 255.
void normalize_chw_scalar(std::span<const std::uint8_t> bgr, std::span<float> chw, int w, int h) noexcept;

#if defined(FF_HAVE_AVX2_TU)
void normalize_chw_avx2(std::span<const std::uint8_t> bgr, std::span<float> chw, int w, int h) noexcept;
void letterbox_resize_avx2(std::span<const std::uint8_t> src, int src_w, int src_h,
                           std::span<std::uint8_t> dst, int model, const Letterbox& lb) noexcept;
/// Resize + normalise in one pass (no intermediate image). Bit-identical to the two-pass path.
void preprocess_fused_avx2(std::span<const std::uint8_t> src, int src_w, int src_h, int model,
                           std::span<float> out, const Letterbox& lb) noexcept;
#endif

/// Runtime-dispatched entry points used by the pipeline.
[[nodiscard]] bool cpu_has_avx2() noexcept;
[[nodiscard]] std::string_view preprocess_backend() noexcept;  // "avx2" or "scalar"
void normalize_chw(std::span<const std::uint8_t> bgr, std::span<float> chw, int w, int h) noexcept;
void letterbox_resize(std::span<const std::uint8_t> src, int src_w, int src_h,
                      std::span<std::uint8_t> dst, int model, const Letterbox& lb) noexcept;

/// Full preprocess: src BGR (src_w x src_h) -> model input CHW float (3*model*model).
/// `scratch` must hold model*model*3 bytes (used only by the two-pass path).
/// Dispatches to the fused AVX2 kernel when available; the two-pass scalar reference otherwise.
void preprocess_frame(std::span<const std::uint8_t> src, int src_w, int src_h, int model,
                      std::span<std::uint8_t> scratch, std::span<float> out, Letterbox& lb) noexcept;
/// Two-pass reference (resize into scratch, then normalise) — used by tests as ground truth.
void preprocess_frame_twopass(std::span<const std::uint8_t> src, int src_w, int src_h, int model,
                              std::span<std::uint8_t> scratch, std::span<float> out, Letterbox& lb) noexcept;

}  // namespace ff
