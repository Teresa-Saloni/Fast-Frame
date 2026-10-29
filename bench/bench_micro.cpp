// Micro-benchmarks for the hot-path primitives. Run:
//   ./build/release/bench/bench_micro --benchmark_min_time=1s
//   FF_FORCE_SCALAR=1 ./build/release/bench/bench_micro --benchmark_filter=Normalize
#include <benchmark/benchmark.h>

#include <mutex>
#include <random>
#include <thread>
#include <vector>
#include <memory>
#include <cstdlib>

#include "fastframe/arena.hpp"
#include "fastframe/preprocess.hpp"
#include "fastframe/spsc_queue.hpp"
#include "fastframe/tracker.hpp"

namespace {

// ---- SPSC queue: ping-pong between two threads, items/sec
void BM_SpscQueue_Throughput(benchmark::State& state) {
  ff::SpscQueue<std::uint64_t, 1024> q;
  const std::uint64_t n = 1'000'000;
  for (auto _ : state) {
    std::thread consumer([&] { std::uint64_t v, c = 0; while (c < n) if (q.try_pop(v)) ++c; });
    for (std::uint64_t i = 0; i < n; ++i) while (!q.try_push(i)) {}
    consumer.join();
  }
  state.SetItemsProcessed(static_cast<std::int64_t>(n) * state.iterations());
}
BENCHMARK(BM_SpscQueue_Throughput)->Unit(benchmark::kMillisecond);

// ---- Baseline to compare against: the same handoff through a mutex.
void BM_MutexQueue_Throughput(benchmark::State& state) {
  std::mutex m;
  std::vector<std::uint64_t> q;
  const std::uint64_t n = 1'000'000;
  for (auto _ : state) {
    q.clear();
    std::thread consumer([&] {
      std::uint64_t c = 0;
      while (c < n) { std::lock_guard lk(m); if (!q.empty()) { q.pop_back(); ++c; } }
    });
    for (std::uint64_t i = 0; i < n; ++i) { for (;;) { std::lock_guard lk(m); if (q.size() < 1024) { q.push_back(i); break; } } }
    consumer.join();
  }
  state.SetItemsProcessed(static_cast<std::int64_t>(n) * state.iterations());
}
BENCHMARK(BM_MutexQueue_Throughput)->Unit(benchmark::kMillisecond);

// ---- Arena acquire/release from N threads
void BM_Arena_AcquireRelease(benchmark::State& state) {
  static ff::FrameArena arena(64, 64);
  for (auto _ : state) {
    auto s = arena.acquire();
    benchmark::DoNotOptimize(s);
    if (s) arena.release(*s);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Arena_AcquireRelease)->Threads(1)->Threads(4)->Threads(8);

// ---- Preprocess kernels on a 640x640 model input
// 64-byte aligned float buffer, like an arena slot.
struct AlignedFloats {
  float* p; std::size_t n;
  explicit AlignedFloats(std::size_t n_) : p(static_cast<float*>(std::aligned_alloc(64, ((n_ * 4 + 63) / 64) * 64))), n(n_) {}
  ~AlignedFloats() { std::free(p); }
  operator std::span<float>() { return {p, n}; }
};

std::vector<std::uint8_t> image(std::size_t n) {
  std::mt19937 rng(1);
  std::vector<std::uint8_t> v(n);
  for (auto& b : v) b = static_cast<std::uint8_t>(rng());
  return v;
}

void BM_Normalize_Scalar(benchmark::State& state) {
  const int w = 640, h = 640;
  const auto img = image(static_cast<std::size_t>(w * h * 3));
  AlignedFloats out(img.size());
  for (auto _ : state) { ff::normalize_chw_scalar(img, out, w, h); benchmark::DoNotOptimize(out.p); }
  state.SetBytesProcessed(static_cast<std::int64_t>(img.size()) * state.iterations());
}
BENCHMARK(BM_Normalize_Scalar)->Unit(benchmark::kMicrosecond);

void BM_Normalize_Dispatched(benchmark::State& state) {
  const int w = 640, h = 640;
  const auto img = image(static_cast<std::size_t>(w * h * 3));
  AlignedFloats out(img.size());
  for (auto _ : state) { ff::normalize_chw(img, out, w, h); benchmark::DoNotOptimize(out.p); }
  state.SetBytesProcessed(static_cast<std::int64_t>(img.size()) * state.iterations());
  state.SetLabel(std::string(ff::preprocess_backend()));
}
BENCHMARK(BM_Normalize_Dispatched)->Unit(benchmark::kMicrosecond);

void BM_Letterbox1080pTo640_Scalar(benchmark::State& state) {
  const auto img = image(1920 * 1080 * 3);
  const auto lb = ff::compute_letterbox(1920, 1080, 640);
  std::vector<std::uint8_t> dst(640 * 640 * 3);
  for (auto _ : state) { ff::letterbox_resize_scalar(img, 1920, 1080, dst, 640, lb); benchmark::DoNotOptimize(dst.data()); }
}
BENCHMARK(BM_Letterbox1080pTo640_Scalar)->Unit(benchmark::kMicrosecond);

void BM_Letterbox1080pTo640_Dispatched(benchmark::State& state) {
  const auto img = image(1920 * 1080 * 3);
  const auto lb = ff::compute_letterbox(1920, 1080, 640);
  std::vector<std::uint8_t> dst(640 * 640 * 3);
  for (auto _ : state) { ff::letterbox_resize(img, 1920, 1080, dst, 640, lb); benchmark::DoNotOptimize(dst.data()); }
  state.SetLabel(std::string(ff::preprocess_backend()));
}
BENCHMARK(BM_Letterbox1080pTo640_Dispatched)->Unit(benchmark::kMicrosecond);

void BM_FullPreprocess1080p(benchmark::State& state) {
  const auto img = image(1920 * 1080 * 3);
  std::vector<std::uint8_t> scratch(640 * 640 * 3);
  AlignedFloats out(3 * 640 * 640);
  ff::Letterbox lb;
  for (auto _ : state) { ff::preprocess_frame(img, 1920, 1080, 640, scratch, out, lb); benchmark::DoNotOptimize(out.p); }
  state.SetLabel(std::string(ff::preprocess_backend()));
}
BENCHMARK(BM_FullPreprocess1080p)->Unit(benchmark::kMicrosecond);

// ---- Tracker with 20 objects
void BM_Tracker20Objects(benchmark::State& state) {
  ff::Tracker t;
  std::vector<ff::Detection> d;
  for (int i = 0; i < 20; ++i) d.push_back({static_cast<float>(i * 50), 0, static_cast<float>(i * 50 + 40), 40, 1.0f, 0});
  for (auto _ : state) {
    for (auto& b : d) { b.x1 += 1; b.x2 += 1; }
    auto out = t.update(d);
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_Tracker20Objects)->Unit(benchmark::kMicrosecond);

}  // namespace
