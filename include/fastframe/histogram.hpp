#pragma once
// Fixed-bucket latency histogram, one per stage thread. No allocation after
// construction; record() runs on the owning thread only (no atomics needed),
// snapshot queries run after the thread has joined.
//
// Buckets are log-linear: 8 sub-buckets per power of two (HdrHistogram-style
// at low precision), which keeps percentile error under ~12 % at any scale.

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>

namespace ff {

class LatencyHistogram {
 public:
  static constexpr int kSub = 8;                 // sub-buckets per octave
  static constexpr int kOctaves = 32;            // up to 2^32 us (~71 min)
  static constexpr int kBuckets = kOctaves * kSub;

  void record_us(std::uint64_t us) noexcept {
    ++buckets_[index_of(us)];
    ++count_;
    sum_us_ += us;
    max_us_ = std::max(max_us_, us);
  }

  [[nodiscard]] std::uint64_t count() const noexcept { return count_; }
  [[nodiscard]] double mean_us() const noexcept {
    return count_ ? static_cast<double>(sum_us_) / static_cast<double>(count_) : 0.0;
  }
  [[nodiscard]] std::uint64_t max_us() const noexcept { return max_us_; }

  /// Upper bound of the bucket holding the p-th percentile (0 < p <= 1).
  [[nodiscard]] std::uint64_t percentile_us(double p) const noexcept {
    if (count_ == 0) return 0;
    const auto target = std::max<std::uint64_t>(1, static_cast<std::uint64_t>(p * static_cast<double>(count_)));
    std::uint64_t acc = 0;
    for (std::size_t b = 0; b < static_cast<std::size_t>(kBuckets); ++b) {
      acc += buckets_[b];
      if (acc >= target) return upper_of(static_cast<int>(b));
    }
    return max_us_;
  }

 private:
  static std::size_t index_of(std::uint64_t us) noexcept {
    if (us < static_cast<std::uint64_t>(kSub)) return static_cast<std::size_t>(us);  // exact for 0..7
    const int oct = 63 - std::countl_zero(us);                     // floor(log2(us)) >= 3
    const auto sub = static_cast<int>((us >> (oct - 3)) & 7u);     // next 3 bits
    const int idx = oct * kSub + sub;
    return static_cast<std::size_t>(std::min(idx, kBuckets - 1));
  }
  static std::uint64_t upper_of(int idx) noexcept {
    if (idx < kSub) return static_cast<std::uint64_t>(idx + 1);
    const int oct = idx / kSub, sub = idx % kSub;
    return static_cast<std::uint64_t>(8 + sub + 1) << static_cast<unsigned>(oct - 3);
  }

  std::array<std::uint64_t, static_cast<std::size_t>(kBuckets)> buckets_{};
  std::uint64_t count_ = 0;
  std::uint64_t sum_us_ = 0;
  std::uint64_t max_us_ = 0;
};

/// RAII timer that records into a histogram on scope exit.
class ScopedTimer {
 public:
  explicit ScopedTimer(LatencyHistogram& h) noexcept : h_(h), t0_(std::chrono::steady_clock::now()) {}
  ~ScopedTimer() {
    const auto dt = std::chrono::steady_clock::now() - t0_;
    h_.record_us(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(dt).count()));
  }
  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;

 private:
  LatencyHistogram& h_;
  std::chrono::steady_clock::time_point t0_;
};

}  // namespace ff
