#include <gtest/gtest.h>

#include <atomic>
#include <vector>

#include "fastframe/thread_pool.hpp"

TEST(ThreadPool, RunsAllTasksAndReturnsValues) {
  ff::ThreadPool pool(4);
  std::vector<std::future<int>> futs;
  for (int i = 0; i < 100; ++i) futs.push_back(pool.submit([i] { return i * i; }));
  for (int i = 0; i < 100; ++i) EXPECT_EQ(futs[static_cast<std::size_t>(i)].get(), i * i);
}

TEST(ThreadPool, DestructorJoinsWithoutLosingQueuedWork) {
  std::atomic<int> done{0};
  {
    ff::ThreadPool pool(2);
    std::vector<std::future<void>> futs;
    for (int i = 0; i < 50; ++i) futs.push_back(pool.submit([&] { done.fetch_add(1); }));
    for (auto& f : futs) f.get();
  }
  EXPECT_EQ(done.load(), 50);
}
