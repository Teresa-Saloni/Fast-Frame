#pragma once
// Pipeline: owns the arena, the four queues and the five stages, wires them
// together, runs to EOS, and reports per-stage metrics.
//
//   Decode -> q0 -> Preprocess -> q1 -> Inference -> q2 -> Tracking -> q3 -> Encode
//
// Decode and Encode are also Stages so the code path is uniform; Decode has
// no input queue (it pulls from the FrameSource) and Encode has no output.

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "fastframe/arena.hpp"
#include "fastframe/io.hpp"
#include "fastframe/stage.hpp"
#include "fastframe/thread_pool.hpp"
#include "fastframe/tracker.hpp"
#include "fastframe/types.hpp"

namespace ff {

struct StageStats {
  std::string name;
  std::uint64_t frames;
  double mean_us;
  std::uint64_t p50_us, p99_us, max_us;
  std::uint64_t stalls;
};

struct PipelineResult {
  std::uint64_t frames_in = 0;
  std::uint64_t frames_out = 0;
  std::uint64_t frames_dropped = 0;
  std::uint64_t arena_exhausted = 0;
  double wall_seconds = 0.0;
  double fps = 0.0;
  std::uint64_t e2e_p50_us = 0, e2e_p99_us = 0;  ///< decode-start to encode-end per frame
  std::vector<StageStats> stages;
  std::string backend_preprocess;
  std::string backend_detector;

  [[nodiscard]] std::string to_json() const;
  [[nodiscard]] std::string to_table() const;
};

class Pipeline {
 public:
  Pipeline(PipelineConfig cfg, std::unique_ptr<FrameSource> src, std::unique_ptr<Detector> det,
           std::unique_ptr<FrameSink> sink);
  ~Pipeline();
  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;

  /// Run until the source is exhausted and every frame has been written. Blocking.
  PipelineResult run();

  [[nodiscard]] const PipelineConfig& config() const noexcept { return cfg_; }
  [[nodiscard]] const FrameArena& arena() const noexcept { return *arena_; }

 private:
  class DecodeStage;
  class PreprocessStage;
  class InferenceStage;
  class TrackingStage;
  class EncodeStage;
  void run_serial();

  PipelineConfig cfg_;
  std::unique_ptr<FrameSource> src_;
  std::unique_ptr<Detector> det_;
  std::unique_ptr<FrameSink> sink_;
  std::unique_ptr<FrameArena> arena_;
  std::unique_ptr<ThreadPool> pool_;
  std::vector<std::unique_ptr<FrameQueue>> queues_;
  std::vector<std::unique_ptr<Stage>> stages_;
  std::vector<std::chrono::steady_clock::time_point> t_in_;  ///< per-slot decode timestamp
  LatencyHistogram e2e_;
};

}  // namespace ff
