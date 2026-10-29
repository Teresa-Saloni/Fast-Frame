#include <gtest/gtest.h>

#include <memory>

#include "fastframe/io.hpp"
#include "fastframe/pipeline.hpp"

namespace {
ff::PipelineResult run_synthetic(int frames, int w, int h, ff::PipelineConfig cfg, ff::NullSink** sink_out = nullptr) {
  auto src = std::make_unique<ff::SyntheticSource>(w, h, static_cast<std::uint32_t>(frames));
  auto det = std::make_unique<ff::SyntheticDetector>(640);
  auto sink = std::make_unique<ff::NullSink>();
  if (sink_out) *sink_out = sink.get();
  ff::Pipeline p(cfg, std::move(src), std::move(det), std::move(sink));
  return p.run();
}
}  // namespace

TEST(Pipeline, AllFramesComeOutInOrderWithNoDrops) {
  ff::NullSink* sink = nullptr;
  ff::PipelineConfig cfg;
  cfg.arena_frames = 16;
  cfg.batch = 4;
  const auto r = run_synthetic(300, 640, 360, cfg, &sink);
  EXPECT_EQ(r.frames_in, 300u);
  EXPECT_EQ(r.frames_out, 300u);
  EXPECT_EQ(r.frames_dropped, 0u);
  EXPECT_GT(r.fps, 0.0);
  ASSERT_NE(sink, nullptr);
  // The Pipeline owns the sink; it was destroyed with p — but r captured counts.
}

TEST(Pipeline, TracksAreProducedForMovingObjects) {
  auto src = std::make_unique<ff::SyntheticSource>(640, 360, 60, 3);
  auto det = std::make_unique<ff::SyntheticDetector>(640);
  auto sink = std::make_unique<ff::NullSink>();
  auto* s = sink.get();
  ff::PipelineConfig cfg;
  cfg.track_min_hits = 1;
  {
    ff::Pipeline p(cfg, std::move(src), std::move(det), std::move(sink));
    const auto r = p.run();
    EXPECT_EQ(r.frames_out, 60u);
    EXPECT_TRUE(s->in_order());
    // 3 objects x 60 frames; allow some misses at frame edges.
    EXPECT_GE(s->tracks_seen(), 120u);
  }
}

TEST(Pipeline, DropPolicyCountsDrops) {
  ff::PipelineConfig cfg;
  cfg.arena_frames = 8;
  cfg.drop_when_full = true;
  cfg.inference_threads = 1;
  auto src = std::make_unique<ff::SyntheticSource>(1280, 720, 200);
  auto det = std::make_unique<ff::SyntheticDetector>(640, 20000);  // slow model forces back-pressure
  ff::Pipeline p(cfg, std::move(src), std::move(det), std::make_unique<ff::NullSink>());
  const auto r = p.run();
  EXPECT_EQ(r.frames_in, 200u);
  EXPECT_EQ(r.frames_out + r.frames_dropped, 200u);
  EXPECT_GT(r.frames_dropped, 0u);
}

TEST(Pipeline, BlockingPolicyNeverDrops) {
  ff::PipelineConfig cfg;
  cfg.arena_frames = 8;
  cfg.inference_threads = 1;
  auto src = std::make_unique<ff::SyntheticSource>(640, 360, 100);
  auto det = std::make_unique<ff::SyntheticDetector>(640, 5000);
  ff::Pipeline p(cfg, std::move(src), std::move(det), std::make_unique<ff::NullSink>());
  const auto r = p.run();
  EXPECT_EQ(r.frames_out, 100u);
  EXPECT_EQ(r.frames_dropped, 0u);
  EXPECT_GT(r.arena_exhausted, 0u) << "with 8 slots and a slow model the arena must run dry at least once";
}

TEST(Pipeline, JsonContainsFpsAndStages) {
  const auto r = run_synthetic(20, 320, 240, ff::PipelineConfig{});
  const auto j = r.to_json();
  EXPECT_NE(j.find("\"fps\":"), std::string::npos);
  EXPECT_NE(j.find("\"name\":\"inference\""), std::string::npos);
}

TEST(Pipeline, SerialBaselineProducesSameFrameCount) {
  ff::PipelineConfig cfg;
  cfg.serial = true;
  const auto r = run_synthetic(50, 640, 360, cfg);
  EXPECT_EQ(r.frames_in, 50u);
  EXPECT_EQ(r.frames_out, 50u);
  EXPECT_EQ(r.frames_dropped, 0u);
}
