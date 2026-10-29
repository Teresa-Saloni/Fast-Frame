#pragma once
// Fixed-size thread pool for the inference stage. A mutex + condition
// variable is the right tool here: tasks are coarse (one batch, tens of ms),
// so lock cost is noise and the simplicity buys correctness. Lock-free is
// reserved for the per-frame hot path (SpscQueue, FrameArena).

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <stop_token>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace ff {

class ThreadPool {
 public:
  explicit ThreadPool(std::size_t n) {
    workers_.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      workers_.emplace_back([this](const std::stop_token& st) { worker(st); });
    }
  }
  ~ThreadPool() {
    for (auto& w : workers_) w.request_stop();
    cv_.notify_all();
    // jthreads join in their destructors (RAII); no explicit join needed.
  }
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  template <typename F>
  auto submit(F&& f) -> std::future<std::invoke_result_t<F>> {
    using R = std::invoke_result_t<F>;
    auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
    std::future<R> fut = task->get_future();
    {
      std::lock_guard lk(mu_);
      tasks_.emplace_back([task] { (*task)(); });
    }
    cv_.notify_one();
    return fut;
  }

  [[nodiscard]] std::size_t size() const noexcept { return workers_.size(); }

 private:
  void worker(const std::stop_token& st) {
    for (;;) {
      std::function<void()> job;
      {
        std::unique_lock lk(mu_);
        cv_.wait(lk, [&] { return st.stop_requested() || !tasks_.empty(); });
        if (tasks_.empty()) {
          if (st.stop_requested()) return;
          continue;
        }
        job = std::move(tasks_.front());
        tasks_.pop_front();
      }
      job();
    }
  }

  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> tasks_;
  std::vector<std::jthread> workers_;
};

}  // namespace ff
