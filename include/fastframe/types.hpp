#pragma once
// Core value types shared by every stage. Everything here is trivially
// copyable so it can travel through lock-free queues by value.

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace ff {

/// Handle to a frame living in the FrameArena. 16 bytes, POD.
/// Queues carry handles, never pixels.
struct FrameHandle {
  std::uint32_t index = 0;  ///< slot in the arena
  std::uint32_t seq = 0;    ///< monotonically increasing frame number
  std::int64_t pts = 0;     ///< presentation timestamp (stream time base)
};
static_assert(std::is_trivially_copyable_v<FrameHandle>);
static_assert(sizeof(FrameHandle) == 16);

/// Sentinel pushed by a producer to tell the consumer the stream ended.
inline constexpr std::uint32_t kEndOfStream = 0xFFFFFFFFu;
inline bool is_eos(const FrameHandle& h) noexcept { return h.index == kEndOfStream; }
inline FrameHandle eos_handle() noexcept { return FrameHandle{kEndOfStream, 0, 0}; }

struct Detection {
  float x1, y1, x2, y2;  ///< pixel coords in the source frame
  float score;
  std::int32_t cls;
};

struct Track {
  std::int32_t id;
  float x1, y1, x2, y2;
  std::int32_t cls;
  std::int32_t age;   ///< frames since creation
  std::int32_t hits;  ///< consecutive matched frames
};

/// Per-frame side data that lives next to the pixel buffer in the arena.
struct FrameMeta {
  std::int32_t width = 0;
  std::int32_t height = 0;
  std::int64_t pts = 0;
  std::uint32_t seq = 0;
  std::vector<Detection> detections;  // filled by Inference
  std::vector<Track> tracks;          // filled by Tracking
};

struct PipelineConfig {
  int width = 1920;
  int height = 1080;
  int model_size = 640;           ///< square model input
  std::uint32_t arena_frames = 64;
  int batch = 4;
  int batch_wait_ms = 8;
  int inference_threads = 2;
  bool pin_threads = false;
  bool serial = false;            ///< baseline mode: every stage on one thread, no queues
  bool drop_when_full = false;    ///< decode drops instead of blocking
  float conf_thresh = 0.25f;
  float nms_iou = 0.45f;
  int track_max_age = 30;
  int track_min_hits = 3;
  float track_iou = 0.3f;
};

}  // namespace ff
