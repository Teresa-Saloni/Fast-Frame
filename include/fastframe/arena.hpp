#pragma once
// FrameArena: one up-front aligned allocation, buffers handed out by index
// through a lock-free Treiber stack. No malloc on the hot path, 64-byte
// aligned frames for AVX2 loads, ABA protected by a 32-bit tag packed with
// the head index into a single 64-bit atomic.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "fastframe/types.hpp"

namespace ff {

class FrameArena {
 public:
  static constexpr std::size_t kAlign = 64;

  /// @param frame_bytes  bytes per pixel buffer (e.g. w*h*3 for BGR)
  /// @param count        number of buffers (<= 2^32-2)
  /// @param model_bytes  extra bytes per slot for the model input tensor (CHW float32)
  FrameArena(std::size_t frame_bytes, std::uint32_t count, std::size_t model_bytes = 0);
  ~FrameArena();
  FrameArena(const FrameArena&) = delete;
  FrameArena& operator=(const FrameArena&) = delete;

  /// Lock-free pop from the free list. Empty -> nullopt (caller decides: wait or drop).
  [[nodiscard]] std::optional<std::uint32_t> acquire() noexcept;
  /// Lock-free push back onto the free list.
  void release(std::uint32_t idx) noexcept;

  [[nodiscard]] std::byte* data(std::uint32_t idx) noexcept { return base_ + idx * stride_; }
  [[nodiscard]] const std::byte* data(std::uint32_t idx) const noexcept { return base_ + idx * stride_; }
  [[nodiscard]] float* model_input(std::uint32_t idx) noexcept {
    return reinterpret_cast<float*>(base_ + idx * stride_ + frame_bytes_aligned_);
  }
  [[nodiscard]] FrameMeta& meta(std::uint32_t idx) noexcept { return meta_[idx]; }

  [[nodiscard]] std::size_t frame_bytes() const noexcept { return frame_bytes_; }
  [[nodiscard]] std::size_t stride() const noexcept { return stride_; }
  [[nodiscard]] std::uint32_t capacity() const noexcept { return count_; }
  /// Approximate free count (racy by design; for metrics only).
  [[nodiscard]] std::uint32_t free_count() const noexcept { return free_count_.load(std::memory_order_relaxed); }
  /// Number of times acquire() found the arena empty.
  [[nodiscard]] std::uint64_t exhausted_count() const noexcept { return exhausted_.load(std::memory_order_relaxed); }

 private:
  static constexpr std::uint32_t kNil = 0xFFFFFFFFu;
  static std::uint64_t pack(std::uint32_t idx, std::uint32_t tag) noexcept {
    return (static_cast<std::uint64_t>(tag) << 32) | idx;
  }
  static std::uint32_t idx_of(std::uint64_t v) noexcept { return static_cast<std::uint32_t>(v); }
  static std::uint32_t tag_of(std::uint64_t v) noexcept { return static_cast<std::uint32_t>(v >> 32); }

  std::byte* base_ = nullptr;
  std::size_t frame_bytes_ = 0;
  std::size_t frame_bytes_aligned_ = 0;
  std::size_t stride_ = 0;
  std::uint32_t count_ = 0;

  // Head lives alone on its cache line; next_[] is written only by the owner
  // of that index between pop and push so it needs no atomics for correctness,
  // but is atomic to keep TSan honest about the publication.
  struct alignas(64) Head { std::atomic<std::uint64_t> v{0}; } head_;
  std::unique_ptr<std::atomic<std::uint32_t>[]> next_;
  std::vector<FrameMeta> meta_;
  alignas(64) std::atomic<std::uint32_t> free_count_{0};
  alignas(64) std::atomic<std::uint64_t> exhausted_{0};
};

}  // namespace ff
