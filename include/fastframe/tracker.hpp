#pragma once
// SORT-style multi-object tracker: constant-velocity Kalman filter per track,
// IoU cost matrix, Hungarian assignment, age/hit bookkeeping. No external
// dependencies; the 7x7 / 4x4 matrix math is written out by hand.

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "fastframe/types.hpp"

namespace ff {

/// Kalman filter with state [cx, cy, s(area), r(aspect), vx, vy, vs] and
/// measurement [cx, cy, s, r] — the SORT parameterisation.
class KalmanBox {
 public:
  static constexpr int kN = 7;  // state size
  static constexpr int kM = 4;  // measurement size
  using Vec7 = std::array<float, kN>;
  using Mat7 = std::array<float, static_cast<std::size_t>(kN) * static_cast<std::size_t>(kN)>;

  explicit KalmanBox(const Detection& d) noexcept;
  void predict() noexcept;
  void update(const Detection& d) noexcept;
  [[nodiscard]] std::array<float, 4> box() const noexcept;  // x1,y1,x2,y2

 private:
  Vec7 x_{};
  Mat7 p_{};
};

struct TrackerParams {
  int max_age = 30;      ///< frames without a match before deletion
  int min_hits = 3;      ///< consecutive matches before a track is reported
  float iou_thresh = 0.3f;
};

class Tracker {
 public:
  using Params = TrackerParams;
  explicit Tracker(Params p = Params{}) noexcept : params_(p) {}

  /// Advance one frame. Returns the tracks that have min_hits (or are freshly matched in warm-up).
  std::vector<Track> update(std::span<const Detection> dets);

  [[nodiscard]] std::size_t active_tracks() const noexcept { return tracks_.size(); }

 private:
  struct Internal {
    KalmanBox kf;
    std::int32_t id;
    std::int32_t cls;
    int age = 0;
    int hits = 0;
    int time_since_update = 0;
    int hit_streak = 0;
  };

  Params params_;
  std::vector<Internal> tracks_;
  std::int32_t next_id_ = 1;
  int frame_count_ = 0;
};

/// IoU between two boxes in (x1,y1,x2,y2).
[[nodiscard]] float iou(float ax1, float ay1, float ax2, float ay2,
                        float bx1, float by1, float bx2, float by2) noexcept;

/// Hungarian algorithm (Kuhn–Munkres), O(n^3), minimising a dense cost matrix
/// of size rows x cols. Returns assignment[row] = col or -1.
std::vector<int> hungarian(const std::vector<float>& cost, int rows, int cols);

/// Greedy non-max suppression on detections, per class. Keeps order by score.
std::vector<Detection> nms(std::vector<Detection> dets, float iou_thresh);

}  // namespace ff
