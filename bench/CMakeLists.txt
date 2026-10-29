add_executable(bench_micro bench_micro.cpp)
target_link_libraries(bench_micro PRIVATE fastframe benchmark::benchmark_main)
target_compile_options(bench_micro PRIVATE -Wall -Wextra)
