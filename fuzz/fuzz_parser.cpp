// libFuzzer harness for the FFR1 container parser.
//   cmake --preset fuzz && cmake --build --preset fuzz
//   ./build/fuzz/fuzz/fuzz_parser fuzz/corpus -max_total_time=600 -max_len=65536
#include <cstddef>
#include <cstdint>
#include <span>

#include "fastframe/frame_header.hpp"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  // Must never crash, hang, or read out of bounds for any input.
  (void)ff::FrameHeaderParser::parse_all(std::span<const std::uint8_t>(data, size));
  return 0;
}
