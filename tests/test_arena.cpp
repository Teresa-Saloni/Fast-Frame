#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include "fastframe/arena.hpp"

TEST(FrameArena, HandsOutEveryBufferOnce) {
  ff::FrameArena a(1000, 16);
  std::vector<bool> seen(16, false);
  for (int i = 0; i < 16; ++i) {
    auto s = a.acquire();
    ASSERT_TRUE(s.has_value());
    ASSERT_LT(*s, 16u);
    EXPECT_FALSE(seen[*s]) << "slot handed out twice";
    seen[*s] = true;
  }
  EXPECT_FALSE(a.acquire().has_value()) << "17th acquire must fail";
  EXPECT_EQ(a.exhausted_count(), 1u);
  a.release(3);
  EXPECT_EQ(a.acquire().value(), 3u);
}

TEST(FrameArena, BuffersAre64ByteAlignedAndDisjoint) {
  ff::FrameArena a(100, 8, 48);  // 100 B frame + 48 B model region, both padded to 64
  EXPECT_EQ(a.stride() % 64, 0u);
  EXPECT_GE(a.stride(), 128u + 64u);
  for (std::uint32_t i = 0; i < 8; ++i) {
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(a.data(i)) % 64, 0u);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(a.model_input(i)) % 64, 0u);
    if (i > 0) EXPECT_EQ(a.data(i) - a.data(i - 1), static_cast<std::ptrdiff_t>(a.stride()));
  }
}

TEST(FrameArena, ConcurrentAcquireReleaseNoDoubleHandout) {
  // 8 threads churn acquire/release; a per-slot owner counter must never exceed 1.
  ff::FrameArena a(64, 32);
  std::vector<std::atomic<int>> owners(32);
  for (auto& o : owners) o.store(0);
  std::atomic<bool> bad{false};
  std::vector<std::thread> ts;
  for (int t = 0; t < 8; ++t) {
    ts.emplace_back([&] {
      for (int i = 0; i < 200'000; ++i) {
        auto s = a.acquire();
        if (!s) continue;
        if (owners[*s].fetch_add(1) != 0) bad = true;
        owners[*s].fetch_sub(1);
        a.release(*s);
      }
    });
  }
  for (auto& t : ts) t.join();
  EXPECT_FALSE(bad.load());
  EXPECT_EQ(a.free_count(), 32u);
}

TEST(FrameArena, RejectsBadCount) {
  EXPECT_THROW(ff::FrameArena(16, 0), std::invalid_argument);
}
