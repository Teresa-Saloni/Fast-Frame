#pragma once
// FastFrame raw-dump container ("FFR1"): the format `fastframe --dump` writes
// for reproducible benchmarks, and what the libFuzzer harness attacks.
//
//   file  := magic "FFR1" | u16 version | u16 flags | u32 width | u32 height
//            | u32 fps_num | u32 fps_den | u32 frame_count | u32 header_crc
//            | frame*
//   frame := u32 size | u64 pts | u8 payload[size]      (size == width*height*3)
//
// Everything is little-endian. The parser never trusts `size`: it is checked
// against width*height*3 and against the remaining bytes before any read.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace ff {

struct FrameHeaderInfo {
  std::uint16_t version;
  std::uint16_t flags;
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t fps_num;
  std::uint32_t fps_den;
  std::uint32_t frame_count;
};

struct FrameRecord {
  std::int64_t pts;
  std::span<const std::uint8_t> payload;
};

class FrameHeaderParser {
 public:
  static constexpr std::size_t kHeaderSize = 4 + 2 + 2 + 4 * 6;
  static constexpr std::size_t kFrameHeaderSize = 4 + 8;
  static constexpr std::uint32_t kMaxDim = 16384;

  enum class Error : std::uint8_t { ok, too_short, bad_magic, bad_version, bad_dims, bad_crc, bad_frame_size, truncated };

  /// Parse the file header. Returns the info or an error; never throws.
  [[nodiscard]] static std::optional<FrameHeaderInfo> parse_header(std::span<const std::uint8_t> buf, Error* err = nullptr) noexcept;

  /// Iterate frames after the header. `offset` is advanced; returns nullopt at end or on error.
  [[nodiscard]] static std::optional<FrameRecord> next_frame(std::span<const std::uint8_t> buf, const FrameHeaderInfo& info,
                                                             std::size_t& offset, Error* err = nullptr) noexcept;

  /// Convenience used by the fuzzer: parse everything, count frames, touch every payload byte.
  [[nodiscard]] static std::size_t parse_all(std::span<const std::uint8_t> buf) noexcept;

  /// Write a header for the given geometry into `out` (must be >= kHeaderSize).
  static void write_header(std::span<std::uint8_t> out, const FrameHeaderInfo& info) noexcept;

  [[nodiscard]] static std::uint32_t crc32(std::span<const std::uint8_t> data) noexcept;
  [[nodiscard]] static std::string_view error_name(Error e) noexcept;
};

}  // namespace ff
