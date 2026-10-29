add_executable(fuzz_parser fuzz_parser.cpp)
target_link_libraries(fuzz_parser PRIVATE fastframe ff_sanitize)
target_link_options(fuzz_parser PRIVATE -fsanitize=fuzzer,address)
