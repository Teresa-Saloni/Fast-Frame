#include "fastframe/tracker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ff {

// ----------------------------------------------------------------- helpers

float iou(float ax1, float ay1, float ax2, float ay2, float bx1, float by1, float bx2, float by2) noexcept {
  const float ix1 = std::max(ax1, bx1), iy1 = std::max(ay1, by1);
  const float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
  const float iw = std::max(0.0f, ix2 - ix1), ih = std::max(0.0f, iy2 - iy1);
  const float inter = iw * ih;
  const float ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter;
  return ua > 0.0f ? inter / ua : 0.0f;
}

namespace {

void box_to_z(const Detection& d, float& cx, float& cy, float& s, float& r) noexcept {
  const float w = d.x2 - d.x1, h = d.y2 - d.y1;
  cx = d.x1 + w * 0.5f;
  cy = d.y1 + h * 0.5f;
  s = w * h;
  r = h > 0.0f ? w / h : 1.0f;
}

using Mat7 = KalmanBox::Mat7;
constexpr int N = KalmanBox::kN;
constexpr int M = KalmanBox::kM;

inline float& at(Mat7& m, int i, int j) noexcept { return m[static_cast<std::size_t>(i * N + j)]; }

// 4x4 inverse by Gauss-Jordan (S is symmetric positive definite here).
bool invert4(std::array<float, 16>& a) noexcept {
  std::array<float, 16> inv{};
  for (int i = 0; i < 4; ++i) inv[static_cast<std::size_t>(i * 4 + i)] = 1.0f;
  for (int c = 0; c < 4; ++c) {
    int piv = c;
    for (int r = c + 1; r < 4; ++r)
      if (std::fabs(a[static_cast<std::size_t>(r * 4 + c)]) > std::fabs(a[static_cast<std::size_t>(piv * 4 + c)])) piv = r;
    if (std::fabs(a[static_cast<std::size_t>(piv * 4 + c)]) < 1e-12f) return false;
    if (piv != c)
      for (int k = 0; k < 4; ++k) {
        std::swap(a[static_cast<std::size_t>(c * 4 + k)], a[static_cast<std::size_t>(piv * 4 + k)]);
        std::swap(inv[static_cast<std::size_t>(c * 4 + k)], inv[static_cast<std::size_t>(piv * 4 + k)]);
      }
    const float d = 1.0f / a[static_cast<std::size_t>(c * 4 + c)];
    for (int k = 0; k < 4; ++k) { a[static_cast<std::size_t>(c * 4 + k)] *= d; inv[static_cast<std::size_t>(c * 4 + k)] *= d; }
    for (int r = 0; r < 4; ++r) {
      if (r == c) continue;
      const float f = a[static_cast<std::size_t>(r * 4 + c)];
      for (int k = 0; k < 4; ++k) {
        a[static_cast<std::size_t>(r * 4 + k)] -= f * a[static_cast<std::size_t>(c * 4 + k)];
        inv[static_cast<std::size_t>(r * 4 + k)] -= f * inv[static_cast<std::size_t>(c * 4 + k)];
      }
    }
  }
  a = inv;
  return true;
}

}  // namespace

// ----------------------------------------------------------------- Kalman

KalmanBox::KalmanBox(const Detection& d) noexcept {
  box_to_z(d, x_[0], x_[1], x_[2], x_[3]);
  // Covariances follow the reference SORT implementation.
  for (int i = 0; i < N; ++i) at(p_, i, i) = i < 4 ? 10.0f : 10000.0f;
}

void KalmanBox::predict() noexcept {
  // x = F x  with F = I + [0 I3; 0 0] (constant velocity on cx, cy, s)
  x_[0] += x_[4];
  x_[1] += x_[5];
  x_[2] += x_[6];
  if (x_[2] < 0.0f) x_[2] = 0.0f;  // area cannot go negative
  // P = F P F^T + Q ; Q = diag(1,1,1,1,0.01,0.01,0.0001)
  Mat7 fp{};
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      float v = at(p_, i, j);
      if (i < 3) v += at(p_, i + 4, j);
      at(fp, i, j) = v;
    }
  Mat7 np{};
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      float v = at(fp, i, j);
      if (j < 3) v += at(fp, i, j + 4);
      at(np, i, j) = v;
    }
  p_ = np;
  static constexpr std::array<float, N> kQ{1.f, 1.f, 1.f, 1.f, 0.01f, 0.01f, 0.0001f};
  for (int i = 0; i < N; ++i) at(p_, i, i) += kQ[static_cast<std::size_t>(i)];
}

void KalmanBox::update(const Detection& d) noexcept {
  float z[M];
  box_to_z(d, z[0], z[1], z[2], z[3]);
  // H = [I4 0]; S = H P H^T + R ; R = diag(1,1,10,10)
  std::array<float, 16> s{};
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < M; ++j) s[static_cast<std::size_t>(i * 4 + j)] = at(p_, i, j);
  s[0] += 1.0f; s[5] += 1.0f; s[10] += 10.0f; s[15] += 10.0f;
  if (!invert4(s)) return;
  // K = P H^T S^-1  (7x4)
  std::array<float, static_cast<std::size_t>(N) * static_cast<std::size_t>(M)> k{};
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < M; ++j) {
      float acc = 0.0f;
      for (int l = 0; l < M; ++l) acc += at(p_, i, l) * s[static_cast<std::size_t>(l * 4 + j)];
      k[static_cast<std::size_t>(i * M + j)] = acc;
    }
  // y = z - H x
  float y[M];
  for (int i = 0; i < M; ++i) y[i] = z[i] - x_[static_cast<std::size_t>(i)];
  // x = x + K y
  for (int i = 0; i < N; ++i) {
    float acc = 0.0f;
    for (int j = 0; j < M; ++j) acc += k[static_cast<std::size_t>(i * M + j)] * y[j];
    x_[static_cast<std::size_t>(i)] += acc;
  }
  // P = (I - K H) P
  Mat7 np{};
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      float acc = at(p_, i, j);
      for (int l = 0; l < M; ++l) acc -= k[static_cast<std::size_t>(i * M + l)] * at(p_, l, j);
      at(np, i, j) = acc;
    }
  p_ = np;
}

std::array<float, 4> KalmanBox::box() const noexcept {
  const float s = std::max(x_[2], 1e-3f), r = std::max(x_[3], 1e-3f);
  const float w = std::sqrt(s * r), h = s / w;
  return {x_[0] - w * 0.5f, x_[1] - h * 0.5f, x_[0] + w * 0.5f, x_[1] + h * 0.5f};
}

// ----------------------------------------------------------------- Hungarian

std::vector<int> hungarian(const std::vector<float>& cost, int rows, int cols) {
  // Square it up by padding with a large cost.
  const int n = std::max(rows, cols);
  constexpr float kInf = std::numeric_limits<float>::max() / 4;
  std::vector<float> a(static_cast<std::size_t>((n + 1) * (n + 1)), 0.0f);
  for (int i = 1; i <= n; ++i)
    for (int j = 1; j <= n; ++j)
      a[static_cast<std::size_t>(i * (n + 1) + j)] =
          (i <= rows && j <= cols) ? cost[static_cast<std::size_t>((i - 1) * cols + (j - 1))] : kInf / 2;

  std::vector<float> u(static_cast<std::size_t>(n + 1)), v(static_cast<std::size_t>(n + 1));
  std::vector<int> p(static_cast<std::size_t>(n + 1)), way(static_cast<std::size_t>(n + 1));
  for (int i = 1; i <= n; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<float> minv(static_cast<std::size_t>(n + 1), kInf);
    std::vector<char> used(static_cast<std::size_t>(n + 1), 0);
    do {
      used[static_cast<std::size_t>(j0)] = 1;
      const int i0 = p[static_cast<std::size_t>(j0)];
      float delta = kInf;
      int j1 = 0;
      for (int j = 1; j <= n; ++j) {
        if (used[static_cast<std::size_t>(j)]) continue;
        const float cur = a[static_cast<std::size_t>(i0 * (n + 1) + j)] - u[static_cast<std::size_t>(i0)] - v[static_cast<std::size_t>(j)];
        if (cur < minv[static_cast<std::size_t>(j)]) { minv[static_cast<std::size_t>(j)] = cur; way[static_cast<std::size_t>(j)] = j0; }
        if (minv[static_cast<std::size_t>(j)] < delta) { delta = minv[static_cast<std::size_t>(j)]; j1 = j; }
      }
      for (int j = 0; j <= n; ++j) {
        if (used[static_cast<std::size_t>(j)]) {
          u[static_cast<std::size_t>(p[static_cast<std::size_t>(j)])] += delta;
          v[static_cast<std::size_t>(j)] -= delta;
        } else {
          minv[static_cast<std::size_t>(j)] -= delta;
        }
      }
      j0 = j1;
    } while (p[static_cast<std::size_t>(j0)] != 0);
    do {
      const int j1 = way[static_cast<std::size_t>(j0)];
      p[static_cast<std::size_t>(j0)] = p[static_cast<std::size_t>(j1)];
      j0 = j1;
    } while (j0 != 0);
  }
  std::vector<int> assign(static_cast<std::size_t>(rows), -1);
  for (int j = 1; j <= n; ++j) {
    const int i = p[static_cast<std::size_t>(j)];
    if (i >= 1 && i <= rows && j <= cols) assign[static_cast<std::size_t>(i - 1)] = j - 1;
  }
  return assign;
}

// ----------------------------------------------------------------- NMS

std::vector<Detection> nms(std::vector<Detection> dets, float iou_thresh) {
  std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) { return a.score > b.score; });
  std::vector<Detection> keep;
  std::vector<char> dead(dets.size(), 0);
  for (std::size_t i = 0; i < dets.size(); ++i) {
    if (dead[i]) continue;
    keep.push_back(dets[i]);
    for (std::size_t j = i + 1; j < dets.size(); ++j) {
      if (dead[j] || dets[j].cls != dets[i].cls) continue;
      if (iou(dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2, dets[j].x1, dets[j].y1, dets[j].x2, dets[j].y2) > iou_thresh)
        dead[j] = 1;
    }
  }
  return keep;
}

// ----------------------------------------------------------------- Tracker

std::vector<Track> Tracker::update(std::span<const Detection> dets) {
  ++frame_count_;
  for (auto& t : tracks_) {
    t.kf.predict();
    ++t.age;
    ++t.time_since_update;
  }

  // Cost = 1 - IoU; entries below threshold are forbidden after assignment.
  const int rows = static_cast<int>(tracks_.size()), cols = static_cast<int>(dets.size());
  std::vector<int> assign(static_cast<std::size_t>(rows), -1);
  if (rows > 0 && cols > 0) {
    std::vector<float> cost(static_cast<std::size_t>(rows * cols));
    for (int i = 0; i < rows; ++i) {
      const auto b = tracks_[static_cast<std::size_t>(i)].kf.box();
      for (int j = 0; j < cols; ++j) {
        const auto& d = dets[static_cast<std::size_t>(j)];
        cost[static_cast<std::size_t>(i * cols + j)] = 1.0f - iou(b[0], b[1], b[2], b[3], d.x1, d.y1, d.x2, d.y2);
      }
    }
    assign = hungarian(cost, rows, cols);
    for (int i = 0; i < rows; ++i) {
      const int j = assign[static_cast<std::size_t>(i)];
      if (j >= 0 && cost[static_cast<std::size_t>(i * cols + j)] > 1.0f - params_.iou_thresh) assign[static_cast<std::size_t>(i)] = -1;
    }
  }

  std::vector<char> det_used(dets.size(), 0);
  for (int i = 0; i < rows; ++i) {
    const int j = assign[static_cast<std::size_t>(i)];
    if (j < 0) continue;
    auto& t = tracks_[static_cast<std::size_t>(i)];
    t.kf.update(dets[static_cast<std::size_t>(j)]);
    t.cls = dets[static_cast<std::size_t>(j)].cls;
    t.time_since_update = 0;
    ++t.hits;
    ++t.hit_streak;
    det_used[static_cast<std::size_t>(j)] = 1;
  }
  for (std::size_t j = 0; j < dets.size(); ++j) {
    if (det_used[j]) continue;
    tracks_.push_back(Internal{KalmanBox(dets[j]), next_id_++, dets[j].cls, 1, 1, 0, 1});
  }

  std::vector<Track> out;
  for (auto& t : tracks_) {
    if (t.time_since_update != 0) t.hit_streak = 0;
    if (t.time_since_update < 1 && (t.hit_streak >= params_.min_hits || frame_count_ <= params_.min_hits)) {
      const auto b = t.kf.box();
      out.push_back(Track{t.id, b[0], b[1], b[2], b[3], t.cls, t.age, t.hits});
    }
  }
  std::erase_if(tracks_, [&](const Internal& t) { return t.time_since_update > params_.max_age; });
  return out;
}

}  // namespace ff
