# FastFrame

A multi-threaded C++20 video-inference pipeline — **decode → preprocess → inference → tracking →
encode** — built to learn performance engineering properly: every stage on its own thread, lock-free
SPSC queues between them, a zero-allocation frame arena, a hand-written AVX2 preprocessing kernel
chosen after profiling, and a CI that runs three sanitizers, a fuzzer, and fails the build if
throughput regresses more than 5 %.

The model is a stock YOLOv8n in ONNX. The project is about how fast and how safely pixels move through it.

```
Decode ──q0──► Preprocess ──q1──► Inference ──q2──► Tracking ──q3──► Encode
(FFmpeg)       (AVX2 fused)      (ONNX Runtime,     (Kalman +        (H.264 / null)
                                  thread pool,       Hungarian)
                                  batch of 4)
            ┌─────────────── frame arena: 64 pre-allocated slots, handles only in queues ───────────────┐
```

## Headline numbers (2-vCPU container, synthetic CI workload; see [BENCHMARKS.md](BENCHMARKS.md))

| | FPS | e2e p99 |
| --- | --- | --- |
| Serial baseline (one thread, no queues, batch 1) | 57.6 | 18 ms |
| Pipelined + batched + AVX2 | **434.5 (7.5x)** | 66 ms |

Micro: fused AVX2 preprocess **2.1x** vs two-pass scalar; lock-free SPSC hand-off **52x** vs mutex queue.

## Build and run (no external deps needed for the synthetic path)

```bash
sudo apt install cmake ninja-build g++-13 libgtest-dev libbenchmark-dev     # Ubuntu 24.04
cmake --preset release && cmake --build --preset release
ctest --preset release                                                      # 8 test binaries, ~2 s
./build/release/fastframe --synthetic 600 --size 1920x1080 --fake-cost 15000 --bench
./build/release/bench/bench_micro
```

Real video + real model:

```bash
sudo apt install libavcodec-dev libavformat-dev libswscale-dev
# ONNX Runtime: download the prebuilt tarball from github.com/microsoft/onnxruntime/releases
export ONNXRUNTIME_ROOT=~/onnxruntime-linux-x64-<version>
pip install ultralytics && python3 scripts/export_yolo.py                   # -> models/yolov8n.onnx
cmake --preset release -DFF_WITH_FFMPEG=ON -DFF_WITH_ORT=ON && cmake --build --preset release
./build/release/fastframe --input clip.mp4 --model models/yolov8n.onnx --output out.mp4 --bench
```

Docker: `docker build -f docker/Dockerfile -t fastframe . && docker run --rm fastframe --synthetic 300 --bench`

## Design decisions

1. **Pipeline parallelism first, data parallelism where it pays.** Stages have different costs, so a
   pipeline keeps every core busy and preserves frame order for free. Only inference — the widest
   stage, measured — gets a thread pool and batching.
2. **SPSC, not MPMC.** Every edge has exactly one producer and one consumer, so a Lamport ring buffer
   with one acquire/release pair per operation is enough; no CAS loops. Head and tail live on separate
   cache lines (`alignas(64)`), and each side caches the other's index to cut coherence traffic.
3. **Handles, not pixels, in queues.** A `FrameHandle` is 16 trivially-copyable bytes. Frames live in
   one `aligned_alloc` arena handed out through a lock-free Treiber stack with a tagged head (ABA-safe).
   Nothing on the hot path calls `malloc`; CI greps for raw `new`/`delete`.
4. **Back-pressure, never silent drops.** Bounded queues (8) block the producer with a
   spin → yield → sleep ladder. Dropping is an explicit policy (`--drop`) and every drop is counted.
5. **Two kernels, one truth.** The AVX2 preprocess (gather-based letterbox fused with
   uint8→float CHW) must be bit-identical to the scalar reference; a test asserts it on 100 random
   images and runs under every sanitizer on both paths (`FF_FORCE_SCALAR=1`).
6. **Only one translation unit sees `-mavx2`.** Dispatch is by `__builtin_cpu_supports` at startup,
   so the same binary runs on a non-AVX2 CPU.
7. **Locks where locks are right.** The inference pool uses a mutex + condition variable: tasks are
   whole batches (tens of ms), so lock cost is noise and simplicity wins.

## Safety and quality

| Check | Where | Status |
| --- | --- | --- |
| ASan / UBSan (gcc) · TSan (clang) | `cmake --preset asan|ubsan|tsan && ctest` | clean (TSan found one real race during development — see BENCHMARKS.md) |
| libFuzzer on the FFR1 container parser | `cmake --preset fuzz`, ClusterFuzzLite nightly | 18 M iterations, 0 crashes |
| clang-tidy `bugprone-*, performance-*, concurrency-*` | `.clang-tidy`, CI | clean, warnings-as-errors |
| `-Wall -Wextra -Wpedantic -Werror -Wconversion -Wshadow …` | every TU, gcc 13 + clang 18 | clean |
| Hardened build (`-fstack-protector-strong -fcf-protection -D_FORTIFY_SOURCE=3 -D_GLIBCXX_ASSERTIONS`) | `cmake --preset hardened` | tests pass |
| CI perf gate | `.github/workflows/ci.yml` → `scripts/check_regression.py` | fails PR on > 5 % FPS drop vs last tag |

## Repository map

```
include/fastframe/   arena, spsc_queue, backoff, thread_pool, histogram, stage, preprocess, tracker,
                     frame_header (fuzz target), io (source/detector/sink interfaces), pipeline
src/                 one .cpp per component; preprocess_avx2.cpp is the only -mavx2 TU;
                     ffmpeg_*.cpp and ort_detector.cpp compile only with FF_WITH_FFMPEG / FF_WITH_ORT
tests/               gtest: queue (10 M-item checksum), arena (8-thread no-double-handout),
                     preprocess (bit-exact scalar vs AVX2), tracker (crossing ids, occlusion), parser, pipeline
bench/               google-benchmark micro + baselines/<tag>.json for the CI gate
fuzz/                libFuzzer harness + seed corpus
scripts/             perf_flamegraph.sh, check_regression.py, make_baseline.py, export_yolo.py, quantize_int8.py
docker/              multi-stage build (toolchain → slim runtime)
.github/workflows/   ci.yml (build matrix, sanitizers, lint+fuzz, perf-gate + history chart), release.yml, fuzz.yml
```

## Profiling

```bash
scripts/perf_flamegraph.sh                     # perf record -g, FlameGraph SVG + perf stat into docs/
```

## Extending it

- **INT8:** `scripts/quantize_int8.py` → `--model models/yolov8n_int8.onnx`; add the FPS/mAP row to BENCHMARKS.md.
- **PGO/LTO:** `cmake --preset pgo-gen`, run the benchmark, then `cmake --preset pgo-use`; `release-lto`.
- **More cameras:** one `Pipeline` per stream sharing a `Detector` (ORT sessions are thread-safe).
- **GPU:** a sixth stage with a pinned staging buffer; the arena already hands out 64-byte-aligned slots.
- **Embedded:** NEON instead of AVX2 behind the same dispatch; static arena sizes; `-fno-exceptions`.

## License

MIT
"# Fast-Frame" 
