#include <gtest/gtest.h>

#include <vector>

#include "fastframe/tracker.hpp"

using ff::Detection;
using ff::Tracker;

TEST(Iou, Basics) {
  EXPECT_FLOAT_EQ(ff::iou(0, 0, 10, 10, 0, 0, 10, 10), 1.0f);
  EXPECT_FLOAT_EQ(ff::iou(0, 0, 10, 10, 10, 10, 20, 20), 0.0f);
  EXPECT_NEAR(ff::iou(0, 0, 10, 10, 5, 0, 15, 10), 50.0f / 150.0f, 1e-6f);
}

TEST(Hungarian, PicksMinimumCostAssignment) {
  // rows=3 cols=3, optimal = (0->1, 1->0, 2->2) = 1 + 2 + 1 = 4
  const std::vector<float> cost = {9, 1, 9, 2, 9, 9, 9, 9, 1};
  const auto a = ff::hungarian(cost, 3, 3);
  EXPECT_EQ(a[0], 1); EXPECT_EQ(a[1], 0); EXPECT_EQ(a[2], 2);
}

TEST(Hungarian, RectangularLeavesRowsUnassigned) {
  const std::vector<float> cost = {1, 9, 9, 1, 5, 5};  // 3 rows x 2 cols
  const auto a = ff::hungarian(cost, 3, 2);
  int assigned = 0;
  for (int v : a) assigned += v >= 0;
  EXPECT_EQ(assigned, 2);
  EXPECT_EQ(a[0], 0); EXPECT_EQ(a[1], 1); EXPECT_EQ(a[2], -1);
}

TEST(Nms, SuppressesOverlapsPerClass) {
  std::vector<Detection> d = {{0, 0, 10, 10, 0.9f, 0}, {1, 1, 11, 11, 0.8f, 0}, {1, 1, 11, 11, 0.7f, 1}, {50, 50, 60, 60, 0.5f, 0}};
  const auto k = ff::nms(d, 0.5f);
  ASSERT_EQ(k.size(), 3u);
  EXPECT_FLOAT_EQ(k[0].score, 0.9f);
}

TEST(Tracker, CrossingObjectsKeepTheirIds) {
  Tracker t(Tracker::Params{30, 3, 0.3f});
  int id_a = -1, id_b = -1;
  for (int f = 0; f < 40; ++f) {
    const float xa = 0.0f + 5.0f * static_cast<float>(f);     // moves right
    const float xb = 200.0f - 5.0f * static_cast<float>(f);   // moves left, same row -> they cross at f=20
    std::vector<Detection> d = {{xa, 0, xa + 20, 20, 1.0f, 0}, {xb, 0, xb + 20, 20, 1.0f, 0}};
    const auto out = t.update(d);
    if (f == 5) { ASSERT_EQ(out.size(), 2u); id_a = out[0].x1 < out[1].x1 ? out[0].id : out[1].id; id_b = id_a == out[0].id ? out[1].id : out[0].id; }
    if (f == 39) {
      ASSERT_EQ(out.size(), 2u);
      // a is now on the right, b on the left; identities must have swapped positions, not ids.
      const auto& right = out[0].x1 > out[1].x1 ? out[0] : out[1];
      const auto& left = out[0].x1 > out[1].x1 ? out[1] : out[0];
      EXPECT_EQ(right.id, id_a);
      EXPECT_EQ(left.id, id_b);
    }
  }
}

TEST(Tracker, SurvivesOcclusionAndDiesAfterMaxAge) {
  Tracker t(Tracker::Params{10, 1, 0.3f});
  const std::vector<Detection> box = {{100, 100, 140, 140, 1.0f, 0}};
  int id = -1;
  for (int f = 0; f < 5; ++f) { auto o = t.update(box); ASSERT_EQ(o.size(), 1u); id = o[0].id; }
  for (int f = 0; f < 8; ++f) t.update(std::vector<Detection>{});   // occluded 8 < max_age
  auto o = t.update(box);
  ASSERT_EQ(o.size(), 1u);
  EXPECT_EQ(o[0].id, id) << "same track must be recovered after a short occlusion";
  for (int f = 0; f < 12; ++f) t.update(std::vector<Detection>{});  // 12 > max_age -> dies
  EXPECT_EQ(t.active_tracks(), 0u);
  o = t.update(box);
  ASSERT_EQ(o.size(), 1u);
  EXPECT_NE(o[0].id, id) << "a new track must get a new id";
}

TEST(Tracker, MinHitsSuppressesFlicker) {
  Tracker t(Tracker::Params{30, 3, 0.3f});
  const std::vector<Detection> box = {{0, 0, 10, 10, 1.0f, 0}};
  for (int f = 0; f < 3; ++f) t.update(box);              // warm-up frames are reported (frame_count <= min_hits)
  t.update(std::vector<Detection>{});                      // one miss resets the streak
  EXPECT_EQ(t.update(box).size(), 0u);                     // streak 1 < 3 -> hidden
  t.update(box);
  EXPECT_EQ(t.update(box).size(), 1u);                     // streak 3 -> shown
}
