#pragma once
// A Stage is one pipeline step running on its own std::jthread, reading
// FrameHandles from an input SPSC queue and writing them to an output one.
// The run loop is generic; subclasses implement process().

#include <concepts>
#include <cstdint>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>

#include "fastframe/backoff.hpp"
#include "fastframe/histogram.hpp"
#include "fastframe/spsc_queue.hpp"
#include "fastframe/types.hpp"

namespace ff {

inline constexpr std::size_t kQueueDepth = 8;
using FrameQueue = SpscQueue<FrameHandle, kQueueDepth>;

/// C++20 concept: anything with process(FrameHandle&) can be wrapped as a stage.
template <typename S>
concept StageLike = requires(S s, FrameHandle& h) {
  { s.process(h) } -> std::same_as<void>;
};

class Stage {
 public:
  explicit Stage(std::string name) : name_(std::move(name)) {}
  virtual ~Stage() = default;
  Stage(const Stage&) = delete;
  Stage& operator=(const Stage&) = delete;

  void connect(FrameQueue* in, FrameQueue* out) noexcept { in_ = in; out_ = out; }

  /// Start the thread. Optionally pin to a CPU core (-1 = no pinning).
  void start(int cpu = -1);
  void request_stop() noexcept { if (thread_.joinable()) thread_.request_stop(); }
  void join() { if (thread_.joinable()) thread_.join(); }

  /// Run process() on the caller's thread (serial baseline mode).
  void process_inline(FrameHandle& h) { ScopedTimer t(hist_); process(h); }
  [[nodiscard]] LatencyHistogram& hist() noexcept { return hist_; }
  [[nodiscard]] std::string_view name() const noexcept { return name_; }
  [[nodiscard]] const LatencyHistogram& histogram() const noexcept { return hist_; }
  [[nodiscard]] std::uint64_t frames() const noexcept { return hist_.count(); }
  [[nodiscard]] std::uint64_t stall_cycles() const noexcept { return stalls_; }

 protected:
  /// Called once per frame on the stage thread.
  virtual void process(FrameHandle& h) = 0;
  /// Called once when EOS is seen, before forwarding it. Flush batches here.
  virtual void on_eos() {}
  /// Push to the output queue, blocking with backoff. Returns false if stopped.
  bool emit(const FrameHandle& h, const std::stop_token& st) noexcept;
  /// Default loop: pop -> process -> emit. Source/batching stages override it.
  virtual void run(const std::stop_token& st);
  [[nodiscard]] FrameQueue* in() noexcept { return in_; }

 private:
  static void pin_to_cpu(int cpu) noexcept;

  std::string name_;
  FrameQueue* in_ = nullptr;
  FrameQueue* out_ = nullptr;
  std::jthread thread_;
  LatencyHistogram hist_;
  std::uint64_t stalls_ = 0;
};

// ----------------------------------------------------------------- inline impl

inline void Stage::start(int cpu) {
  thread_ = std::jthread([this, cpu](const std::stop_token& st) {
    if (cpu >= 0) pin_to_cpu(cpu);
    run(st);
  });
}

inline bool Stage::emit(const FrameHandle& h, const std::stop_token& st) noexcept {
  if (out_ == nullptr) return true;
  Backoff bo;
  while (!out_->try_push(h)) {
    if (st.stop_requested()) return false;
    ++stalls_;
    bo.pause();
  }
  return true;
}

inline void Stage::run(const std::stop_token& st) {
  Backoff bo;
  FrameHandle h;
  while (!st.stop_requested()) {
    if (!in_->try_pop(h)) {
      bo.pause();
      continue;
    }
    bo.reset();
    if (is_eos(h)) {
      on_eos();
      (void)emit(h, st);
      return;
    }
    {
      ScopedTimer t(hist_);
      process(h);
    }
    if (!emit(h, st)) return;
  }
}

}  // namespace ff
