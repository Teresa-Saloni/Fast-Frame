#pragma once
// Spin -> yield -> sleep ladder for threads waiting on a lock-free queue.
// Spinning with `pause` keeps latency low when the other side is about to
// act; yielding and sleeping stop a starved pipeline from burning all cores.

#include <chrono>
#include <cstdint>
#include <thread>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define FF_CPU_PAUSE() _mm_pause()
#elif defined(__aarch64__)
#define FF_CPU_PAUSE() asm volatile("yield" ::: "memory")
#else
#define FF_CPU_PAUSE() ((void)0)
#endif

namespace ff {

class Backoff {
 public:
  explicit Backoff(std::uint32_t spin_limit = 64, std::uint32_t yield_limit = 16) noexcept
      : spin_limit_(spin_limit), yield_limit_(yield_limit) {}

  void pause() noexcept {
    if (n_ < spin_limit_) {
      FF_CPU_PAUSE();
    } else if (n_ < spin_limit_ + yield_limit_) {
      std::this_thread::yield();
    } else {
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    ++n_;
  }
  void reset() noexcept { n_ = 0; }

 private:
  std::uint32_t n_ = 0;
  std::uint32_t spin_limit_;
  std::uint32_t yield_limit_;
};

}  // namespace ff
