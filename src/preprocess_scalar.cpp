#include <algorithm>
#include <cassert>
#include <cstring>

#include "fastframe/preprocess.hpp"

namespace ff {

Letterbox compute_letterbox(int src_w, int src_h, int model) noexcept {
  const float r = std::min(static_cast<float>(model) / static_cast<float>(src_w),
                           static_cast<float>(model) / static_cast<float>(src_h));
  Letterbox lb;
  lb.scale = r;
  lb.new_w = std::max(1, static_cast<int>(static_cast<float>(src_w) * r));
  lb.new_h = std::max(1, static_cast<int>(static_cast<float>(src_h) * r));
  lb.pad_x = (model - lb.new_w) / 2;
  lb.pad_y = (model - lb.new_h) / 2;
  return lb;
}

void letterbox_resize_scalar(std::span<const std::uint8_t> src, int src_w, int src_h,
                             std::span<std::uint8_t> dst, int model, const Letterbox& lb) noexcept {
  assert(src.size() >= static_cast<std::size_t>(src_w) * static_cast<std::size_t>(src_h) * 3);
  assert(dst.size() >= static_cast<std::size_t>(model) * static_cast<std::size_t>(model) * 3);
  const auto mw = static_cast<std::size_t>(model);
  std::memset(dst.data(), 114, mw * mw * 3);

  // Fixed-point source coordinate: x_src = (x_dst * src_w) / new_w  (integer, same as AVX2 path)
  for (int y = 0; y < lb.new_h; ++y) {
    const int sy = std::min(src_h - 1, static_cast<int>((static_cast<std::int64_t>(y) * src_h) / lb.new_h));
    const std::uint8_t* srow = src.data() + static_cast<std::size_t>(sy) * static_cast<std::size_t>(src_w) * 3;
    std::uint8_t* drow = dst.data() + (static_cast<std::size_t>(y + lb.pad_y) * mw + static_cast<std::size_t>(lb.pad_x)) * 3;
    for (int x = 0; x < lb.new_w; ++x) {
      const int sx = std::min(src_w - 1, static_cast<int>((static_cast<std::int64_t>(x) * src_w) / lb.new_w));
      const std::uint8_t* p = srow + static_cast<std::size_t>(sx) * 3;
      drow[static_cast<std::size_t>(x) * 3 + 0] = p[0];
      drow[static_cast<std::size_t>(x) * 3 + 1] = p[1];
      drow[static_cast<std::size_t>(x) * 3 + 2] = p[2];
    }
  }
}

void normalize_chw_scalar(std::span<const std::uint8_t> bgr, std::span<float> chw, int w, int h) noexcept {
  const auto plane = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
  assert(bgr.size() >= plane * 3 && chw.size() >= plane * 3);
  constexpr float kScale = 1.0f / 255.0f;
  float* r = chw.data();
  float* g = chw.data() + plane;
  float* b = chw.data() + 2 * plane;
  const std::uint8_t* p = bgr.data();
  for (std::size_t i = 0; i < plane; ++i, p += 3) {
    b[i] = static_cast<float>(p[0]) * kScale;
    g[i] = static_cast<float>(p[1]) * kScale;
    r[i] = static_cast<float>(p[2]) * kScale;
  }
}

}  // namespace ff
