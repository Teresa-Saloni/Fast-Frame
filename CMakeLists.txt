cmake_minimum_required(VERSION 3.25)
project(fastframe VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------- options
option(FF_ENABLE_AVX2   "Build the AVX2 preprocessing kernel (runtime-dispatched)" ON)
option(FF_WITH_FFMPEG   "Build the FFmpeg decode/encode backend"                   OFF)
option(FF_WITH_ORT      "Build the ONNX Runtime detector backend"                  OFF)
option(FF_BUILD_TESTS   "Build unit tests (gtest)"                                 ON)
option(FF_BUILD_BENCH   "Build micro-benchmarks (google-benchmark)"                ON)
option(FF_BUILD_FUZZ    "Build libFuzzer harness (clang only)"                     OFF)
option(FF_FETCH_DEPS    "Fetch gtest/benchmark with FetchContent if not found"     ON)
option(FF_HARDENED      "Enable hardening flags (stack protector, CET, FORTIFY)"   OFF)
option(FF_LTO           "Enable link-time optimisation in Release"                 OFF)
set(FF_SANITIZE "" CACHE STRING "address | thread | undefined | fuzzer")

# ---------------------------------------------------------------- warnings
add_library(ff_warnings INTERFACE)
target_compile_options(ff_warnings INTERFACE
  -Wall -Wextra -Wpedantic -Werror -Wshadow -Wconversion -Wsign-conversion
  -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wunused -Woverloaded-virtual
  -Wnull-dereference -Wdouble-promotion -Wformat=2)

# ---------------------------------------------------------------- sanitizers
add_library(ff_sanitize INTERFACE)
if(FF_SANITIZE STREQUAL "address")
  target_compile_options(ff_sanitize INTERFACE -fsanitize=address -fno-omit-frame-pointer -O1 -g)
  target_link_options(ff_sanitize INTERFACE -fsanitize=address)
elseif(FF_SANITIZE STREQUAL "thread")
  target_compile_options(ff_sanitize INTERFACE -fsanitize=thread -O1 -g)
  target_link_options(ff_sanitize INTERFACE -fsanitize=thread)
elseif(FF_SANITIZE STREQUAL "undefined")
  target_compile_options(ff_sanitize INTERFACE -fsanitize=undefined -fno-sanitize-recover=all -O1 -g)
  target_link_options(ff_sanitize INTERFACE -fsanitize=undefined)
elseif(FF_SANITIZE STREQUAL "fuzzer")
  target_compile_options(ff_sanitize INTERFACE -fsanitize=fuzzer-no-link,address -O1 -g)
  target_link_options(ff_sanitize INTERFACE -fsanitize=address)
endif()

# ---------------------------------------------------------------- hardening
add_library(ff_hardened INTERFACE)
if(FF_HARDENED)
  target_compile_options(ff_hardened INTERFACE
    -fstack-protector-strong -fcf-protection=full -fstack-clash-protection
    -D_FORTIFY_SOURCE=3 -D_GLIBCXX_ASSERTIONS)
  target_link_options(ff_hardened INTERFACE -Wl,-z,relro,-z,now)
endif()

if(FF_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_ok OUTPUT ipo_msg)
  if(ipo_ok)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
  endif()
endif()

find_package(Threads REQUIRED)

# ---------------------------------------------------------------- core library
add_library(fastframe
  src/arena.cpp
  src/preprocess.cpp
  src/preprocess_scalar.cpp
  src/tracker.cpp
  src/frame_header.cpp
  src/pipeline.cpp
  src/synthetic.cpp)
target_include_directories(fastframe PUBLIC include)
target_link_libraries(fastframe PUBLIC Threads::Threads PRIVATE ff_warnings ff_sanitize ff_hardened)
target_compile_definitions(fastframe PUBLIC FF_VERSION="${PROJECT_VERSION}")

if(FF_ENABLE_AVX2)
  # Only this translation unit gets -mavx2; the binary stays runnable on
  # non-AVX2 CPUs because dispatch happens at runtime via __builtin_cpu_supports.
  target_sources(fastframe PRIVATE src/preprocess_avx2.cpp)
  set_source_files_properties(src/preprocess_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
  target_compile_definitions(fastframe PUBLIC FF_HAVE_AVX2_TU=1)
endif()

if(FF_WITH_FFMPEG)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET libavcodec libavformat libavutil libswscale)
  target_sources(fastframe PRIVATE src/ffmpeg_source.cpp src/ffmpeg_sink.cpp)
  target_link_libraries(fastframe PUBLIC PkgConfig::FFMPEG)
  target_compile_definitions(fastframe PUBLIC FF_WITH_FFMPEG=1)
endif()

if(FF_WITH_ORT)
  find_package(ONNXRuntime REQUIRED)   # cmake/FindONNXRuntime.cmake
  target_sources(fastframe PRIVATE src/ort_detector.cpp)
  target_link_libraries(fastframe PUBLIC ONNXRuntime::ONNXRuntime)
  target_compile_definitions(fastframe PUBLIC FF_WITH_ORT=1)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---------------------------------------------------------------- CLI
add_executable(fastframe_cli src/main.cpp)
set_target_properties(fastframe_cli PROPERTIES OUTPUT_NAME fastframe)
target_link_libraries(fastframe_cli PRIVATE fastframe ff_warnings ff_sanitize ff_hardened)

# ---------------------------------------------------------------- deps for tests/bench
include(FetchContent)
if(FF_BUILD_TESTS)
  find_package(GTest QUIET)
  if(NOT GTest_FOUND AND FF_FETCH_DEPS)
    FetchContent_Declare(googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
      URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()
  enable_testing()
  add_subdirectory(tests)
endif()

if(FF_BUILD_BENCH)
  find_package(benchmark QUIET)
  if(NOT benchmark_FOUND AND FF_FETCH_DEPS)
    FetchContent_Declare(benchmark
      URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.tar.gz
      URL_HASH SHA256=32131c08ee31eeff2c8968d7e874f3cb648034377dfc32a4c377fa8796d84981)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)
  endif()
  add_subdirectory(bench)
endif()

if(FF_BUILD_FUZZ)
  add_subdirectory(fuzz)
endif()

install(TARGETS fastframe_cli RUNTIME DESTINATION bin)
