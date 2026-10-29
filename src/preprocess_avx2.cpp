// This translation unit is the ONLY one compiled with -mavx2 -mfma.
// Entry points are called only after cpu_has_avx2() returned true.

#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>

#include "fastframe/preprocess.hpp"

namespace ff {

// Interleaved BGR bytes -> 8 floats per channel per iteration.
//
// For 8 pixels (24 bytes) we take two 16-byte loads: v0 at byte 0 (pixels 0-4)
// and v1 at byte 12 (pixels 4-8). pshufb runs per 128-bit lane, so the same
// mask pulls bytes {0,3,6,9} = B of pixels 0..3 from v0 and B of pixels 4..7
// from v1. unpacklo_epi32 glues those two dwords into 8 bytes, cvtepu8_epi32
// widens to 8 x int32, cvtepi32_ps to float, mul by 1/255, store.
void normalize_chw_avx2(std::span<const std::uint8_t> bgr, std::span<float> chw, int w, int h) noexcept {
  const auto plane = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
  assert(bgr.size() >= plane * 3 && chw.size() >= plane * 3);

  const __m128i mask_b = _mm_setr_epi8(0, 3, 6, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
  const __m128i mask_g = _mm_setr_epi8(1, 4, 7, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
  const __m128i mask_r = _mm_setr_epi8(2, 5, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
  const __m256 scale = _mm256_set1_ps(1.0f / 255.0f);

  float* r = chw.data();
  float* g = chw.data() + plane;
  float* b = chw.data() + 2 * plane;
  const std::uint8_t* p = bgr.data();

  std::size_t i = 0;
  // The second load reads bytes [3i+12, 3i+28); keep it inside the buffer.
  for (; i + 10 <= plane; i += 8, p += 24) {
    const __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    const __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 12));
    const __m128i b8 = _mm_unpacklo_epi32(_mm_shuffle_epi8(v0, mask_b), _mm_shuffle_epi8(v1, mask_b));
    const __m128i g8 = _mm_unpacklo_epi32(_mm_shuffle_epi8(v0, mask_g), _mm_shuffle_epi8(v1, mask_g));
    const __m128i r8 = _mm_unpacklo_epi32(_mm_shuffle_epi8(v0, mask_r), _mm_shuffle_epi8(v1, mask_r));
    _mm256_storeu_ps(b + i, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b8)), scale));
    _mm256_storeu_ps(g + i, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g8)), scale));
    _mm256_storeu_ps(r + i, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(r8)), scale));
  }
  // Scalar tail: identical arithmetic (uint8 -> float * (1/255)) so results match bit-for-bit.
  constexpr float kScale = 1.0f / 255.0f;
  for (; i < plane; ++i, p += 3) {
    b[i] = static_cast<float>(p[0]) * kScale;
    g[i] = static_cast<float>(p[1]) * kScale;
    r[i] = static_cast<float>(p[2]) * kScale;
  }
}

// Nearest-neighbour letterbox resize using a per-row column LUT and 8-wide
// dword gathers. Each gather pulls 4 bytes (B,G,R,+1 junk) per source pixel;
// a lane-local pshufb compacts 4 dwords to 12 bytes, stored with a 16-byte
// store that overlaps the next pixel group (overwritten by the next store).
void letterbox_resize_avx2(std::span<const std::uint8_t> src, int src_w, int src_h,
                           std::span<std::uint8_t> dst, int model, const Letterbox& lb) noexcept {
  assert(src.size() >= static_cast<std::size_t>(src_w) * static_cast<std::size_t>(src_h) * 3);
  assert(dst.size() >= static_cast<std::size_t>(model) * static_cast<std::size_t>(model) * 3);
  const auto mw = static_cast<std::size_t>(model);
  std::memset(dst.data(), 114, mw * mw * 3);

  // Column LUT in byte offsets (3*sx), computed once per call (new_w <= model).
  static thread_local std::vector<int> lut;
  lut.resize(static_cast<std::size_t>(lb.new_w));
  for (int x = 0; x < lb.new_w; ++x) {
    const int sx = std::min(src_w - 1, static_cast<int>((static_cast<std::int64_t>(x) * src_w) / lb.new_w));
    lut[static_cast<std::size_t>(x)] = sx * 3;
  }

  const __m256i compact = _mm256_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1,
                                           0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);
  const std::size_t src_bytes = src.size();

  for (int y = 0; y < lb.new_h; ++y) {
    const int sy = std::min(src_h - 1, static_cast<int>((static_cast<std::int64_t>(y) * src_h) / lb.new_h));
    const std::uint8_t* srow = src.data() + static_cast<std::size_t>(sy) * static_cast<std::size_t>(src_w) * 3;
    std::uint8_t* drow = dst.data() + (static_cast<std::size_t>(y + lb.pad_y) * mw + static_cast<std::size_t>(lb.pad_x)) * 3;
    const std::size_t row_off = static_cast<std::size_t>(srow - src.data());

    int x = 0;
    // Guard: gather reads 4 bytes per pixel and the 16-byte store overlaps 4
    // bytes past the group; both must stay inside their buffers.
    for (; x + 8 <= lb.new_w; x += 8) {
      const int last_off = lut[static_cast<std::size_t>(x + 7)];
      if (row_off + static_cast<std::size_t>(last_off) + 4 > src_bytes) break;        // last source row edge
      if (static_cast<std::size_t>(x) * 3 + 32 > mw * 3 - static_cast<std::size_t>(lb.pad_x) * 3) break;  // dst row edge
      const __m256i idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lut.data() + x));
      const __m256i px = _mm256_i32gather_epi32(reinterpret_cast<const int*>(srow), idx, 1);
      const __m256i packed = _mm256_shuffle_epi8(px, compact);
      std::uint8_t* d = drow + static_cast<std::size_t>(x) * 3;
      _mm_storeu_si128(reinterpret_cast<__m128i*>(d), _mm256_castsi256_si128(packed));
      _mm_storeu_si128(reinterpret_cast<__m128i*>(d + 12), _mm256_extracti128_si256(packed, 1));
    }
    const int vec_end = x;
    for (; x < lb.new_w; ++x) {
      const std::uint8_t* p = srow + static_cast<std::size_t>(lut[static_cast<std::size_t>(x)]);
      std::uint8_t* d = drow + static_cast<std::size_t>(x) * 3;
      d[0] = p[0]; d[1] = p[1]; d[2] = p[2];
    }
    // The last vector store spilled 4 bytes past the group; if no scalar
    // pixel overwrote them they belong to the right padding -> restore 114.
    if (vec_end > 0 && vec_end == lb.new_w) std::memset(drow + static_cast<std::size_t>(vec_end) * 3, 114, 4);
  }
}


// Fused letterbox-resize + normalise: gathers source pixels and writes the
// float CHW planes directly, skipping the 1.2 MB intermediate BGR image
// (one write + one read less per frame). Output is bit-identical to the
// two-pass path: same nearest-neighbour index math, same uint8 -> float * (1/255).
void preprocess_fused_avx2(std::span<const std::uint8_t> src, int src_w, int src_h, int model,
                           std::span<float> out, const Letterbox& lb) noexcept {
  const auto m = static_cast<std::size_t>(model);
  const std::size_t plane = m * m;
  assert(out.size() >= plane * 3);
  float* rp = out.data();
  float* gp = out.data() + plane;
  float* bp = out.data() + 2 * plane;

  constexpr float kPad = 114.0f * (1.0f / 255.0f);
  const __m256 pad8 = _mm256_set1_ps(kPad);
  const __m256 scale = _mm256_set1_ps(1.0f / 255.0f);
  // Per-lane masks: dword k holds {B,G,R,x} of pixel k -> pull byte 0/1/2 of each dword.
  const __m256i mask_b = _mm256_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                          0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
  const __m256i mask_g = _mm256_setr_epi8(1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                          1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
  const __m256i mask_r = _mm256_setr_epi8(2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                          2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

  static thread_local std::vector<int> lut;
  lut.resize(static_cast<std::size_t>(lb.new_w));
  for (int x = 0; x < lb.new_w; ++x) {
    const int sx = std::min(src_w - 1, static_cast<int>((static_cast<std::int64_t>(x) * src_w) / lb.new_w));
    lut[static_cast<std::size_t>(x)] = sx * 3;
  }
  const std::size_t src_bytes = src.size();

  auto fill_pad = [&](std::size_t from, std::size_t count) {
    std::size_t k = 0;
    for (; k + 8 <= count; k += 8) {
      _mm256_storeu_ps(rp + from + k, pad8); _mm256_storeu_ps(gp + from + k, pad8); _mm256_storeu_ps(bp + from + k, pad8);
    }
    for (; k < count; ++k) { rp[from + k] = kPad; gp[from + k] = kPad; bp[from + k] = kPad; }
  };

  // Top padding rows.
  fill_pad(0, static_cast<std::size_t>(lb.pad_y) * m);
  for (int y = 0; y < lb.new_h; ++y) {
    const int sy = std::min(src_h - 1, static_cast<int>((static_cast<std::int64_t>(y) * src_h) / lb.new_h));
    const std::uint8_t* srow = src.data() + static_cast<std::size_t>(sy) * static_cast<std::size_t>(src_w) * 3;
    const std::size_t row_off = static_cast<std::size_t>(srow - src.data());
    const std::size_t orow = (static_cast<std::size_t>(y + lb.pad_y)) * m;
    fill_pad(orow, static_cast<std::size_t>(lb.pad_x));
    const std::size_t o0 = orow + static_cast<std::size_t>(lb.pad_x);
    int x = 0;
    for (; x + 8 <= lb.new_w; x += 8) {
      const int last_off = lut[static_cast<std::size_t>(x + 7)];
      if (row_off + static_cast<std::size_t>(last_off) + 4 > src_bytes) break;  // 4-byte gather past the buffer end
      const __m256i idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lut.data() + x));
      const __m256i px = _mm256_i32gather_epi32(reinterpret_cast<const int*>(srow), idx, 1);
      // shuffle gives 4 bytes per lane; permute lanes' low dwords together -> 8 bytes in the low xmm.
      const __m256i sb = _mm256_shuffle_epi8(px, mask_b), sg = _mm256_shuffle_epi8(px, mask_g), sr = _mm256_shuffle_epi8(px, mask_r);
      const __m128i b8 = _mm_unpacklo_epi32(_mm256_castsi256_si128(sb), _mm256_extracti128_si256(sb, 1));
      const __m128i g8 = _mm_unpacklo_epi32(_mm256_castsi256_si128(sg), _mm256_extracti128_si256(sg, 1));
      const __m128i r8 = _mm_unpacklo_epi32(_mm256_castsi256_si128(sr), _mm256_extracti128_si256(sr, 1));
      const std::size_t o = o0 + static_cast<std::size_t>(x);
      _mm256_storeu_ps(bp + o, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b8)), scale));
      _mm256_storeu_ps(gp + o, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g8)), scale));
      _mm256_storeu_ps(rp + o, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(r8)), scale));
    }
    constexpr float kScale = 1.0f / 255.0f;
    for (; x < lb.new_w; ++x) {
      const std::uint8_t* q = srow + static_cast<std::size_t>(lut[static_cast<std::size_t>(x)]);
      const std::size_t o = o0 + static_cast<std::size_t>(x);
      bp[o] = static_cast<float>(q[0]) * kScale;
      gp[o] = static_cast<float>(q[1]) * kScale;
      rp[o] = static_cast<float>(q[2]) * kScale;
    }
    fill_pad(o0 + static_cast<std::size_t>(lb.new_w), m - static_cast<std::size_t>(lb.pad_x) - static_cast<std::size_t>(lb.new_w));
  }
  // Bottom padding rows.
  const std::size_t bottom_from = (static_cast<std::size_t>(lb.pad_y) + static_cast<std::size_t>(lb.new_h)) * m;
  fill_pad(bottom_from, plane - bottom_from);
}

}  // namespace ff
