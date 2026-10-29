#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
#include <thread>

#include "fastframe/io.hpp"
#include "fastframe/tracker.hpp"

namespace ff {

// ----------------------------------------------------------------- SyntheticSource

SyntheticSource::SyntheticSource(int w, int h, std::uint32_t frames, int objects, std::uint32_t seed)
    : w_(w), h_(h), frames_(frames) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> px(0.1f, 0.6f), vel(-3.0f, 3.0f), sz(0.06f, 0.14f);
  std::uniform_int_distribution<int> col(80, 255);
  for (int i = 0; i < objects; ++i) {
    Obj o;
    o.x = px(rng) * static_cast<float>(w);
    o.y = px(rng) * static_cast<float>(h);
    o.vx = vel(rng);
    o.vy = vel(rng);
    o.size = sz(rng) * static_cast<float>(std::min(w, h));
    // Bright, saturated, non-grey colours so the synthetic detector can threshold them.
    o.b = static_cast<std::uint8_t>(i % 3 == 0 ? col(rng) : 0);
    o.g = static_cast<std::uint8_t>(i % 3 == 1 ? col(rng) : 0);
    o.r = static_cast<std::uint8_t>(i % 3 == 2 ? col(rng) : 0);
    if (o.b == 0 && o.g == 0 && o.r == 0) o.r = 200;
    objs_.push_back(o);
  }
}

std::vector<Detection> SyntheticSource::truth(std::uint32_t frame) const noexcept {
  std::vector<Detection> out;
  for (const auto& o : objs_) {
    // Objects bounce; replay their motion deterministically for frame `frame`.
    float x = o.x, y = o.y, vx = o.vx, vy = o.vy;
    for (std::uint32_t f = 0; f < frame; ++f) {
      x += vx; y += vy;
      if (x < 0 || x + o.size > static_cast<float>(w_)) { vx = -vx; x += 2 * vx; }
      if (y < 0 || y + o.size > static_cast<float>(h_)) { vy = -vy; y += 2 * vy; }
    }
    out.push_back(Detection{x, y, x + o.size, y + o.size, 1.0f, 0});
  }
  return out;
}

bool SyntheticSource::next(std::span<std::uint8_t> dst, std::int64_t& pts) {
  if (produced_ >= frames_) return false;
  const auto w = static_cast<std::size_t>(w_);
  std::memset(dst.data(), 40, w * static_cast<std::size_t>(h_) * 3);  // dark grey background
  const auto boxes = truth(produced_);
  for (std::size_t i = 0; i < boxes.size(); ++i) {
    const auto& b = boxes[i];
    const auto& o = objs_[i];
    const int x1 = std::max(0, static_cast<int>(b.x1)), y1 = std::max(0, static_cast<int>(b.y1));
    const int x2 = std::min(w_, static_cast<int>(b.x2)), y2 = std::min(h_, static_cast<int>(b.y2));
    for (int y = y1; y < y2; ++y) {
      std::uint8_t* row = dst.data() + (static_cast<std::size_t>(y) * w + static_cast<std::size_t>(x1)) * 3;
      for (int x = x1; x < x2; ++x, row += 3) { row[0] = o.b; row[1] = o.g; row[2] = o.r; }
    }
  }
  pts = static_cast<std::int64_t>(produced_) * 33;  // ~30 fps in ms
  ++produced_;
  return true;
}

// ----------------------------------------------------------------- SyntheticDetector

void SyntheticDetector::infer(std::span<const float* const> inputs, std::span<std::vector<Detection>> outputs,
                              float conf_thresh, float nms_iou) {
  const auto s = static_cast<std::size_t>(size_);
  const std::size_t plane = s * s;
  for (std::size_t n = 0; n < inputs.size(); ++n) {
    const float* r = inputs[n];
    const float* g = r + plane;
    const float* b = r + 2 * plane;
    // Connected components on a coarse 8x8 grid of "saturated" cells.
    const std::size_t cell = 8, gw = s / cell;
    std::vector<std::uint8_t> hot(gw * gw, 0);
    for (std::size_t gy = 0; gy < gw; ++gy)
      for (std::size_t gx = 0; gx < gw; ++gx) {
        const std::size_t i = (gy * cell + cell / 2) * s + gx * cell + cell / 2;
        const float mx = std::max({r[i], g[i], b[i]}), mn = std::min({r[i], g[i], b[i]});
        hot[gy * gw + gx] = (mx > 0.3f && mx - mn > 0.25f) ? 1 : 0;
      }
    std::vector<Detection> dets;
    std::vector<std::uint8_t> seen(gw * gw, 0);
    std::vector<std::size_t> stack;
    for (std::size_t start = 0; start < gw * gw; ++start) {
      if (!hot[start] || seen[start]) continue;
      std::size_t minx = gw, miny = gw, maxx = 0, maxy = 0, count = 0;
      stack.assign(1, start);
      seen[start] = 1;
      while (!stack.empty()) {
        const std::size_t c = stack.back(); stack.pop_back();
        const std::size_t cx = c % gw, cy = c / gw;
        minx = std::min(minx, cx); maxx = std::max(maxx, cx);
        miny = std::min(miny, cy); maxy = std::max(maxy, cy);
        ++count;
        const std::size_t nb[4] = {c + 1, c - 1, c + gw, c - gw};
        for (const std::size_t q : nb) {
          if (q >= gw * gw || seen[q] || !hot[q]) continue;
          if ((q == c + 1 && cx + 1 >= gw) || (q == c - 1 && cx == 0)) continue;
          seen[q] = 1;
          stack.push_back(q);
        }
      }
      if (count < 2) continue;
      Detection d;
      d.x1 = static_cast<float>(minx * cell);
      d.y1 = static_cast<float>(miny * cell);
      d.x2 = static_cast<float>((maxx + 1) * cell);
      d.y2 = static_cast<float>((maxy + 1) * cell);
      d.score = std::min(1.0f, 0.5f + 0.02f * static_cast<float>(count));
      d.cls = 0;
      if (d.score >= conf_thresh) dets.push_back(d);
    }
    outputs[n] = nms(std::move(dets), nms_iou);
  }
  if (cost_us_ > 0) {
    // Emulate a real model: spin (not sleep) so it shows up in perf like inference would.
    const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(cost_us_);
    while (std::chrono::steady_clock::now() < until) { }
  }
}

// ----------------------------------------------------------------- NullSink / drawing

void NullSink::write(std::span<const std::uint8_t> /*bgr*/, int /*w*/, int /*h*/, std::int64_t pts,
                     std::span<const Track> tracks) {
  if (pts < last_pts_) in_order_ = false;
  last_pts_ = pts;
  ++frames_;
  tracks_ += tracks.size();
}

void draw_tracks(std::span<std::uint8_t> bgr, int w, int h, std::span<const Track> tracks) noexcept {
  const auto stride = static_cast<std::size_t>(w) * 3;
  for (const auto& t : tracks) {
    const int x1 = std::clamp(static_cast<int>(t.x1), 0, w - 1), y1 = std::clamp(static_cast<int>(t.y1), 0, h - 1);
    const int x2 = std::clamp(static_cast<int>(t.x2), 0, w - 1), y2 = std::clamp(static_cast<int>(t.y2), 0, h - 1);
    // Colour from id so a track keeps its colour across frames.
    const std::uint8_t cb = static_cast<std::uint8_t>(37 * t.id), cg = static_cast<std::uint8_t>(91 * t.id + 60),
                       cr = static_cast<std::uint8_t>(151 * t.id + 120);
    auto put = [&](int x, int y) {
      std::uint8_t* p = bgr.data() + static_cast<std::size_t>(y) * stride + static_cast<std::size_t>(x) * 3;
      p[0] = cb; p[1] = cg; p[2] = cr;
    };
    for (int x = x1; x <= x2; ++x) { put(x, y1); put(x, y2); if (y1 + 1 < h) put(x, y1 + 1); if (y2 - 1 >= 0) put(x, y2 - 1); }
    for (int y = y1; y <= y2; ++y) { put(x1, y); put(x2, y); if (x1 + 1 < w) put(x1 + 1, y); if (x2 - 1 >= 0) put(x2 - 1, y); }
    // Tiny id tick marks: `id` short bars above the box.
    for (int k = 0; k < std::min(t.id, 20); ++k)
      for (int dx = 0; dx < 3; ++dx) {
        const int x = x1 + k * 5 + dx, y = y1 - 4;
        if (x < w && y >= 0) put(x, y);
      }
  }
}

}  // namespace ff
