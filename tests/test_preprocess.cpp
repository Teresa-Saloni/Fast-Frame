#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

#include "fastframe/preprocess.hpp"

namespace {
std::vector<std::uint8_t> random_image(std::size_t n, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<std::uint8_t> v(n);
  for (auto& b : v) b = static_cast<std::uint8_t>(rng());
  return v;
}
}  // namespace

TEST(Preprocess, LetterboxGeometry) {
  const auto lb = ff::compute_letterbox(1920, 1080, 640);
  EXPECT_EQ(lb.new_w, 640);
  EXPECT_EQ(lb.new_h, 360);
  EXPECT_EQ(lb.pad_x, 0);
  EXPECT_EQ(lb.pad_y, 140);
  EXPECT_FLOAT_EQ(lb.scale, 640.0f / 1920.0f);
}

TEST(Preprocess, NormalizeScalarMatchesDispatchedKernelBitExact) {
  // 100 random images of odd sizes; the dispatched path (AVX2 when available,
  // scalar under FF_FORCE_SCALAR) must match the scalar reference exactly.
  for (std::uint32_t seed = 0; seed < 100; ++seed) {
    const int w = 7 + static_cast<int>(seed % 23) * 3, h = 5 + static_cast<int>(seed % 11);
    const auto plane = static_cast<std::size_t>(w * h);
    const auto img = random_image(plane * 3, seed);
    std::vector<float> ref(plane * 3), out(plane * 3, -1.0f);
    ff::normalize_chw_scalar(img, ref, w, h);
    ff::normalize_chw(img, out, w, h);
    for (std::size_t i = 0; i < plane * 3; ++i) ASSERT_EQ(ref[i], out[i]) << "seed " << seed << " idx " << i;
  }
}

TEST(Preprocess, NormalizeValuesAreInUnitRangeAndChannelSwapped) {
  const int w = 16, h = 2;
  std::vector<std::uint8_t> img(static_cast<std::size_t>(w * h * 3));
  for (std::size_t i = 0; i < img.size(); i += 3) { img[i] = 255; img[i + 1] = 0; img[i + 2] = 128; }  // B,G,R
  std::vector<float> out(img.size());
  ff::normalize_chw(img, out, w, h);
  EXPECT_FLOAT_EQ(out[0], 128.0f / 255.0f);          // R plane first
  EXPECT_FLOAT_EQ(out[static_cast<std::size_t>(w * h)], 0.0f);
  EXPECT_FLOAT_EQ(out[static_cast<std::size_t>(2 * w * h)], 1.0f);
}

TEST(Preprocess, LetterboxResizeScalarMatchesDispatched) {
  const int model = 64;
  const std::vector<std::pair<int, int>> sizes = {{64, 64}, {128, 72}, {33, 97}, {1, 1}, {200, 20}, {7, 300}};
  for (std::size_t k = 0; k < sizes.size(); ++k) {
    const auto [w, h] = sizes[k];
    const auto img = random_image(static_cast<std::size_t>(w * h * 3), static_cast<std::uint32_t>(k));
    const auto lb = ff::compute_letterbox(w, h, model);
    std::vector<std::uint8_t> a(static_cast<std::size_t>(model * model * 3)), b(a.size());
    ff::letterbox_resize_scalar(img, w, h, a, model, lb);
    ff::letterbox_resize(img, w, h, b, model, lb);
    ASSERT_EQ(a, b) << "size " << w << "x" << h;
  }
}

TEST(Preprocess, PaddingIs114) {
  const auto img = random_image(1920 * 1080 * 3, 7);
  ff::Letterbox lb;
  std::vector<std::uint8_t> scratch(640 * 640 * 3);
  std::vector<float> out(3 * 640 * 640);
  ff::preprocess_frame(img, 1920, 1080, 640, scratch, out, lb);
  EXPECT_FLOAT_EQ(out[0], 114.0f / 255.0f);                                // top-left pad, R plane
  EXPECT_FLOAT_EQ(out[3 * 640 * 640 - 1], 114.0f / 255.0f);                // bottom-right pad, B plane

  EXPECT_FLOAT_EQ(out[static_cast<std::size_t>(lb.pad_y - 1) * 640 + 639], 114.0f / 255.0f);
}

TEST(Preprocess, FusedDispatchedMatchesTwoPassReferenceBitExact) {
  const int model = 96;
  const std::vector<std::pair<int, int>> sizes = {{96, 96}, {192, 108}, {33, 97}, {1, 1}, {400, 20}, {7, 300}, {1920, 1080}};
  for (std::size_t k = 0; k < sizes.size(); ++k) {
    const auto [w, h] = sizes[k];
    const auto img = random_image(static_cast<std::size_t>(w * h * 3), 100 + static_cast<std::uint32_t>(k));
    std::vector<std::uint8_t> scratch(static_cast<std::size_t>(model * model * 3));
    std::vector<float> ref(static_cast<std::size_t>(3 * model * model)), out(ref.size(), -7.0f);
    ff::Letterbox a, b;
    ff::preprocess_frame_twopass(img, w, h, model, scratch, ref, a);
    ff::preprocess_frame(img, w, h, model, scratch, out, b);
    EXPECT_EQ(a.pad_x, b.pad_x);
    for (std::size_t i = 0; i < ref.size(); ++i) ASSERT_EQ(ref[i], out[i]) << "size " << w << "x" << h << " idx " << i;
  }
}
