#pragma once
// Single-producer / single-consumer bounded ring buffer (Lamport, 1983) with
// C++20 atomics. Wait-free for one producer and one consumer.
//
// Memory ordering, in one breath: the producer writes the slot then
// *releases* tail; the consumer *acquires* tail before reading the slot, so
// the slot write happens-before the slot read. Symmetrically for head.
//
// head_ and tail_ sit on separate cache lines (alignas(64)) so the producer
// and consumer never false-share. Each side also caches the other side's
// index locally and only reloads it when the cached value says full/empty,
// which cuts cross-core traffic further (see BENCHMARKS.md).

#include <array>
#include <atomic>
#include <cstddef>
#include <type_traits>

namespace ff {

template <typename T, std::size_t N>
class SpscQueue {
  static_assert(N >= 2 && (N & (N - 1)) == 0, "N must be a power of two");
  static_assert(std::is_trivially_copyable_v<T>, "queue payload must be trivially copyable");

 public:
  SpscQueue() = default;
  SpscQueue(const SpscQueue&) = delete;
  SpscQueue& operator=(const SpscQueue&) = delete;

  /// Producer side. Returns false when full (no blocking here; the Stage decides).
  [[nodiscard]] bool try_push(const T& v) noexcept {
    const std::size_t t = tail_.load(std::memory_order_relaxed);
    if (t - head_cache_ == N) {                               // looks full
      head_cache_ = head_.load(std::memory_order_acquire);    // refresh once
      if (t - head_cache_ == N) return false;                 // really full
    }
    buf_[t & kMask] = v;
    tail_.store(t + 1, std::memory_order_release);
    return true;
  }

  /// Consumer side. Returns false when empty.
  [[nodiscard]] bool try_pop(T& out) noexcept {
    const std::size_t h = head_.load(std::memory_order_relaxed);
    if (h == tail_cache_) {
      tail_cache_ = tail_.load(std::memory_order_acquire);
      if (h == tail_cache_) return false;
    }
    out = buf_[h & kMask];
    head_.store(h + 1, std::memory_order_release);
    return true;
  }

  /// Approximate occupancy for metrics (racy by design).
  [[nodiscard]] std::size_t size_approx() const noexcept {
    return tail_.load(std::memory_order_relaxed) - head_.load(std::memory_order_relaxed);
  }
  static constexpr std::size_t capacity() noexcept { return N; }

  // C++20 atomic wait/notify hooks for the blocking variant (used by Backoff::wait_push/pop).
  void wait_not_full(std::size_t observed_head) noexcept { head_.wait(observed_head, std::memory_order_acquire); }
  void wait_not_empty(std::size_t observed_tail) noexcept { tail_.wait(observed_tail, std::memory_order_acquire); }
  void notify_producer() noexcept { head_.notify_one(); }
  void notify_consumer() noexcept { tail_.notify_one(); }
  [[nodiscard]] std::size_t head_snapshot() const noexcept { return head_.load(std::memory_order_acquire); }
  [[nodiscard]] std::size_t tail_snapshot() const noexcept { return tail_.load(std::memory_order_acquire); }

 private:
  static constexpr std::size_t kMask = N - 1;

  alignas(64) std::atomic<std::size_t> head_{0};  // consumer-owned
  alignas(64) std::size_t tail_cache_{0};         // consumer's private copy of tail
  alignas(64) std::atomic<std::size_t> tail_{0};  // producer-owned
  alignas(64) std::size_t head_cache_{0};         // producer's private copy of head
  alignas(64) std::array<T, N> buf_{};
};

}  // namespace ff
