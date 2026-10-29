#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "fastframe/frame_header.hpp"

using P = ff::FrameHeaderParser;

namespace {
std::vector<std::uint8_t> make_file(std::uint32_t w, std::uint32_t h, std::uint32_t frames) {
  std::vector<std::uint8_t> buf(P::kHeaderSize);
  P::write_header(buf, ff::FrameHeaderInfo{1, 0, w, h, 30, 1, frames});
  const std::uint32_t size = w * h * 3;
  for (std::uint32_t f = 0; f < frames; ++f) {
    for (int i = 0; i < 4; ++i) buf.push_back(static_cast<std::uint8_t>(size >> (8 * i)));
    for (int i = 0; i < 8; ++i) buf.push_back(static_cast<std::uint8_t>((f * 33ull) >> (8 * i)));
    buf.insert(buf.end(), size, static_cast<std::uint8_t>(f));
  }
  return buf;
}
}  // namespace

TEST(FrameHeader, RoundTrip) {
  const auto buf = make_file(4, 3, 5);
  P::Error e;
  const auto h = P::parse_header(buf, &e);
  ASSERT_TRUE(h.has_value()) << P::error_name(e);
  EXPECT_EQ(h->width, 4u); EXPECT_EQ(h->height, 3u); EXPECT_EQ(h->frame_count, 5u);
  EXPECT_EQ(P::parse_all(buf), 5u);
  std::size_t off = 0;
  const auto f = P::next_frame(buf, *h, off, &e);
  ASSERT_TRUE(f.has_value());
  EXPECT_EQ(f->pts, 0);
  EXPECT_EQ(f->payload.size(), 36u);
}

TEST(FrameHeader, RejectsBadMagicVersionDimsCrc) {
  auto buf = make_file(2, 2, 1);
  P::Error e;
  auto bad = buf; bad[0] = 'X';
  EXPECT_FALSE(P::parse_header(bad, &e)); EXPECT_EQ(e, P::Error::bad_magic);
  bad = buf; bad[4] = 2;
  EXPECT_FALSE(P::parse_header(bad, &e)); EXPECT_EQ(e, P::Error::bad_version);
  bad = buf; bad[8] = 0; bad[9] = 0; bad[10] = 0; bad[11] = 0;
  EXPECT_FALSE(P::parse_header(bad, &e)); EXPECT_EQ(e, P::Error::bad_dims);
  bad = buf; bad[16] ^= 1;  // fps_num flips -> crc mismatch
  EXPECT_FALSE(P::parse_header(bad, &e)); EXPECT_EQ(e, P::Error::bad_crc);
}

TEST(FrameHeader, LyingSizeFieldNeverReadsOutOfBounds) {
  auto buf = make_file(2, 2, 2);
  // Claim the first frame is 4 GB.
  buf[P::kHeaderSize + 0] = 0xFF; buf[P::kHeaderSize + 1] = 0xFF; buf[P::kHeaderSize + 2] = 0xFF; buf[P::kHeaderSize + 3] = 0xFF;
  EXPECT_EQ(P::parse_all(buf), 0u);
  // Truncate the second frame's payload.
  auto t = make_file(2, 2, 2);
  t.resize(t.size() - 5);
  EXPECT_EQ(P::parse_all(t), 1u);
}

TEST(FrameHeader, EmptyAndTinyInputs) {
  EXPECT_EQ(P::parse_all(std::vector<std::uint8_t>{}), 0u);
  EXPECT_EQ(P::parse_all(std::vector<std::uint8_t>(3, 0)), 0u);
  EXPECT_EQ(P::parse_all(std::vector<std::uint8_t>(P::kHeaderSize, 0)), 0u);
}
