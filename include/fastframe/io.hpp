#pragma once
// Abstract source / detector / sink so the pipeline can run with synthetic
// backends (tests, CI, sanitizers) or real ones (FFmpeg, ONNX Runtime).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "fastframe/preprocess.hpp"
#include "fastframe/types.hpp"

namespace ff {

/// Produces BGR uint8 HWC frames into caller-provided buffers.
class FrameSource {
 public:
  virtual ~FrameSource() = default;
  [[nodiscard]] virtual int width() const noexcept = 0;
  [[nodiscard]] virtual int height() const noexcept = 0;
  /// Fill `dst` (width*height*3 bytes). Returns false at end of stream. Sets pts.
  virtual bool next(std::span<std::uint8_t> dst, std::int64_t& pts) = 0;
};

/// Runs a batch of preprocessed CHW tensors and writes detections in model space.
class Detector {
 public:
  virtual ~Detector() = default;
  [[nodiscard]] virtual int input_size() const noexcept = 0;  // e.g. 640
  /// inputs[i] points to 3*S*S floats; outputs[i] receives detections in model pixel space.
  virtual void infer(std::span<const float* const> inputs, std::span<std::vector<Detection>> outputs,
                     float conf_thresh, float nms_iou) = 0;
  [[nodiscard]] virtual std::string name() const = 0;
};

/// Consumes finished frames (with tracks drawn) — file encoder or /dev/null.
class FrameSink {
 public:
  virtual ~FrameSink() = default;
  virtual void write(std::span<const std::uint8_t> bgr, int w, int h, std::int64_t pts,
                     std::span<const Track> tracks) = 0;
  virtual void finish() {}
};

// ----------------------------------------------------------------- synthetic backends

/// Deterministic moving-pattern generator: `n` frames of w x h with k moving squares.
class SyntheticSource final : public FrameSource {
 public:
  SyntheticSource(int w, int h, std::uint32_t frames, int objects = 4, std::uint32_t seed = 42);
  int width() const noexcept override { return w_; }
  int height() const noexcept override { return h_; }
  bool next(std::span<std::uint8_t> dst, std::int64_t& pts) override;
  /// Ground-truth boxes for frame `i`, in source pixels (used by tests).
  [[nodiscard]] std::vector<Detection> truth(std::uint32_t frame) const noexcept;

 private:
  struct Obj { float x, y, vx, vy, size; std::uint8_t b, g, r; };
  int w_, h_;
  std::uint32_t frames_, produced_ = 0;
  std::vector<Obj> objs_;
};

/// Detector that "sees" the synthetic squares by colour thresholding — zero
/// ML, runs anywhere, gives the pipeline real boxes to track. Costs ~1 ms,
/// configurable with `fake_cost_us` to emulate a real model's latency.
class SyntheticDetector final : public Detector {
 public:
  explicit SyntheticDetector(int size = 640, int fake_cost_us = 0) : size_(size), cost_us_(fake_cost_us) {}
  int input_size() const noexcept override { return size_; }
  void infer(std::span<const float* const> inputs, std::span<std::vector<Detection>> outputs,
             float conf_thresh, float nms_iou) override;
  std::string name() const override { return "synthetic"; }

 private:
  int size_;
  int cost_us_;
};

/// Sink that counts frames and verifies in-order delivery.
class NullSink final : public FrameSink {
 public:
  void write(std::span<const std::uint8_t> bgr, int w, int h, std::int64_t pts, std::span<const Track> tracks) override;
  [[nodiscard]] std::uint64_t frames() const noexcept { return frames_; }
  [[nodiscard]] std::uint64_t tracks_seen() const noexcept { return tracks_; }
  [[nodiscard]] bool in_order() const noexcept { return in_order_; }

 private:
  std::uint64_t frames_ = 0, tracks_ = 0;
  std::int64_t last_pts_ = -1;
  bool in_order_ = true;
};

/// Draw track boxes + ids into a BGR frame (shared by sinks).
void draw_tracks(std::span<std::uint8_t> bgr, int w, int h, std::span<const Track> tracks) noexcept;

#if defined(FF_WITH_FFMPEG)
std::unique_ptr<FrameSource> open_ffmpeg_source(const std::string& path);
std::unique_ptr<FrameSink> open_ffmpeg_sink(const std::string& path, int w, int h, int fps);
#endif
#if defined(FF_WITH_ORT)
std::unique_ptr<Detector> open_ort_detector(const std::string& model_path, int intra_threads);
#endif

}  // namespace ff
