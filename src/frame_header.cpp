#include "fastframe/frame_header.hpp"

#include <array>
#include <cstring>

namespace ff {

namespace {
std::uint16_t rd16(const std::uint8_t* p) noexcept { return static_cast<std::uint16_t>(p[0] | (p[1] << 8)); }
std::uint32_t rd32(const std::uint8_t* p) noexcept {
  return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}
std::uint64_t rd64(const std::uint8_t* p) noexcept {
  return static_cast<std::uint64_t>(rd32(p)) | (static_cast<std::uint64_t>(rd32(p + 4)) << 32);
}
void wr16(std::uint8_t* p, std::uint16_t v) noexcept { p[0] = static_cast<std::uint8_t>(v); p[1] = static_cast<std::uint8_t>(v >> 8); }
void wr32(std::uint8_t* p, std::uint32_t v) noexcept {
  for (int i = 0; i < 4; ++i) p[i] = static_cast<std::uint8_t>(v >> (8 * i));
}
void set_err(FrameHeaderParser::Error* e, FrameHeaderParser::Error v) noexcept { if (e != nullptr) *e = v; }
}  // namespace

std::uint32_t FrameHeaderParser::crc32(std::span<const std::uint8_t> data) noexcept {
  static const auto table = [] {
    std::array<std::uint32_t, 256> t{};
    for (std::uint32_t i = 0; i < 256; ++i) {
      std::uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      t[i] = c;
    }
    return t;
  }();
  std::uint32_t c = 0xFFFFFFFFu;
  for (const std::uint8_t b : data) c = table[(c ^ b) & 0xFFu] ^ (c >> 8);
  return c ^ 0xFFFFFFFFu;
}

std::optional<FrameHeaderInfo> FrameHeaderParser::parse_header(std::span<const std::uint8_t> buf, Error* err) noexcept {
  set_err(err, Error::ok);
  if (buf.size() < kHeaderSize) { set_err(err, Error::too_short); return std::nullopt; }
  const std::uint8_t* p = buf.data();
  if (std::memcmp(p, "FFR1", 4) != 0) { set_err(err, Error::bad_magic); return std::nullopt; }
  FrameHeaderInfo h;
  h.version = rd16(p + 4);
  h.flags = rd16(p + 6);
  h.width = rd32(p + 8);
  h.height = rd32(p + 12);
  h.fps_num = rd32(p + 16);
  h.fps_den = rd32(p + 20);
  h.frame_count = rd32(p + 24);
  const std::uint32_t crc = rd32(p + 28);
  if (h.version != 1) { set_err(err, Error::bad_version); return std::nullopt; }
  if (h.width == 0 || h.height == 0 || h.width > kMaxDim || h.height > kMaxDim || h.fps_den == 0) {
    set_err(err, Error::bad_dims);
    return std::nullopt;
  }
  if (crc32(buf.subspan(0, 28)) != crc) { set_err(err, Error::bad_crc); return std::nullopt; }
  return h;
}

std::optional<FrameRecord> FrameHeaderParser::next_frame(std::span<const std::uint8_t> buf, const FrameHeaderInfo& info,
                                                         std::size_t& offset, Error* err) noexcept {
  set_err(err, Error::ok);
  if (offset < kHeaderSize) offset = kHeaderSize;
  if (offset >= buf.size()) return std::nullopt;  // clean end
  if (buf.size() - offset < kFrameHeaderSize) { set_err(err, Error::truncated); return std::nullopt; }
  const std::uint8_t* p = buf.data() + offset;
  const std::uint32_t size = rd32(p);
  const auto pts = static_cast<std::int64_t>(rd64(p + 4));
  // width*height*3 fits in u64; compare without overflow in u32.
  const std::uint64_t expected = static_cast<std::uint64_t>(info.width) * info.height * 3u;
  if (size != expected) { set_err(err, Error::bad_frame_size); return std::nullopt; }
  const std::size_t body = offset + kFrameHeaderSize;
  if (buf.size() - body < size) { set_err(err, Error::truncated); return std::nullopt; }
  offset = body + size;
  return FrameRecord{pts, buf.subspan(body, size)};
}

std::size_t FrameHeaderParser::parse_all(std::span<const std::uint8_t> buf) noexcept {
  Error e = Error::ok;
  const auto h = parse_header(buf, &e);
  if (!h) return 0;
  std::size_t off = kHeaderSize, n = 0;
  volatile std::uint32_t sink = 0;  // force the payload bytes to be touched
  while (auto f = next_frame(buf, *h, off, &e)) {
    std::uint32_t acc = 0;
    for (const std::uint8_t b : f->payload) acc += b;
    sink = acc;
    ++n;
  }
  (void)sink;
  return n;
}

void FrameHeaderParser::write_header(std::span<std::uint8_t> out, const FrameHeaderInfo& info) noexcept {
  std::uint8_t* p = out.data();
  p[0] = 'F'; p[1] = 'F'; p[2] = 'R'; p[3] = '1';
  wr16(p + 4, info.version);
  wr16(p + 6, info.flags);
  wr32(p + 8, info.width);
  wr32(p + 12, info.height);
  wr32(p + 16, info.fps_num);
  wr32(p + 20, info.fps_den);
  wr32(p + 24, info.frame_count);
  wr32(p + 28, crc32(std::span<const std::uint8_t>(p, 28)));
}

std::string_view FrameHeaderParser::error_name(Error e) noexcept {
  switch (e) {
    case Error::ok: return "ok";
    case Error::too_short: return "too_short";
    case Error::bad_magic: return "bad_magic";
    case Error::bad_version: return "bad_version";
    case Error::bad_dims: return "bad_dims";
    case Error::bad_crc: return "bad_crc";
    case Error::bad_frame_size: return "bad_frame_size";
    case Error::truncated: return "truncated";
  }
  return "?";
}

}  // namespace ff
