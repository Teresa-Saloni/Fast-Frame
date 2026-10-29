#include "fastframe/arena.hpp"

#include <cstdlib>
#include <new>
#include <stdexcept>

namespace ff {

namespace {
std::size_t round_up(std::size_t v, std::size_t a) noexcept { return (v + a - 1) / a * a; }
}  // namespace

FrameArena::FrameArena(std::size_t frame_bytes, std::uint32_t count, std::size_t model_bytes)
    : frame_bytes_(frame_bytes),
      frame_bytes_aligned_(round_up(frame_bytes, kAlign)),
      stride_(round_up(frame_bytes_aligned_ + model_bytes, kAlign)),
      count_(count),
      next_(new std::atomic<std::uint32_t>[count]),
      meta_(count) {
  if (count == 0 || count >= kNil) throw std::invalid_argument("FrameArena: bad count");
  const std::size_t total = stride_ * count_;
  base_ = static_cast<std::byte*>(std::aligned_alloc(kAlign, total));
  if (base_ == nullptr) throw std::bad_alloc();

  // Build the free list: 0 -> 1 -> 2 -> ... -> count-1 -> nil
  for (std::uint32_t i = 0; i < count_; ++i) {
    next_[i].store(i + 1 < count_ ? i + 1 : kNil, std::memory_order_relaxed);
  }
  head_.v.store(pack(0, 0), std::memory_order_release);
  free_count_.store(count_, std::memory_order_relaxed);
}

FrameArena::~FrameArena() { std::free(base_); }

std::optional<std::uint32_t> FrameArena::acquire() noexcept {
  std::uint64_t old = head_.v.load(std::memory_order_acquire);
  for (;;) {
    const std::uint32_t idx = idx_of(old);
    if (idx == kNil) {
      exhausted_.fetch_add(1, std::memory_order_relaxed);
      return std::nullopt;
    }
    const std::uint32_t nxt = next_[idx].load(std::memory_order_relaxed);
    // Tag increments on every successful pop, so a recycled index with the
    // same value as `old` still fails the CAS (classic ABA fix).
    const std::uint64_t desired = pack(nxt, tag_of(old) + 1);
    if (head_.v.compare_exchange_weak(old, desired, std::memory_order_acq_rel, std::memory_order_acquire)) {
      free_count_.fetch_sub(1, std::memory_order_relaxed);
      return idx;
    }
  }
}

void FrameArena::release(std::uint32_t idx) noexcept {
  std::uint64_t old = head_.v.load(std::memory_order_acquire);
  for (;;) {
    next_[idx].store(idx_of(old), std::memory_order_relaxed);
    const std::uint64_t desired = pack(idx, tag_of(old) + 1);
    if (head_.v.compare_exchange_weak(old, desired, std::memory_order_acq_rel, std::memory_order_acquire)) {
      free_count_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
  }
}

}  // namespace ff
