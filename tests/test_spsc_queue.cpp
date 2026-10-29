#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <thread>

#include "fastframe/spsc_queue.hpp"

using ff::SpscQueue;

TEST(SpscQueue, FifoOrderSingleThread) {
  SpscQueue<int, 8> q;
  for (int i = 0; i < 8; ++i) EXPECT_TRUE(q.try_push(i));
  EXPECT_FALSE(q.try_push(99)) << "queue must report full at capacity";
  int v = -1;
  for (int i = 0; i < 8; ++i) { ASSERT_TRUE(q.try_pop(v)); EXPECT_EQ(v, i); }
  EXPECT_FALSE(q.try_pop(v)) << "queue must report empty";
}

TEST(SpscQueue, WrapsAroundManyTimes) {
  SpscQueue<std::uint32_t, 4> q;
  std::uint32_t v = 0;
  for (std::uint32_t i = 0; i < 10'000; ++i) {
    ASSERT_TRUE(q.try_push(i));
    ASSERT_TRUE(q.try_pop(v));
    EXPECT_EQ(v, i);
  }
}

TEST(SpscQueue, ProducerConsumerChecksum) {
  // 10M items across two threads; the consumer must see every item exactly once, in order.
  constexpr std::uint64_t kN = 10'000'000;
  SpscQueue<std::uint64_t, 1024> q;
  std::atomic<bool> ok{true};
  std::thread consumer([&] {
    std::uint64_t expect = 0, v = 0;
    while (expect < kN) {
      if (!q.try_pop(v)) continue;
      if (v != expect) { ok = false; return; }
      ++expect;
    }
  });
  for (std::uint64_t i = 0; i < kN; ++i) while (!q.try_push(i)) {}
  consumer.join();
  EXPECT_TRUE(ok.load());
}

TEST(SpscQueue, HeadAndTailOnSeparateCacheLines) {
  // Structural guarantee behind the false-sharing claim.
  EXPECT_GE(sizeof(SpscQueue<int, 2>), 5 * 64u);
  EXPECT_EQ(alignof(SpscQueue<int, 2>), 64u);
}
