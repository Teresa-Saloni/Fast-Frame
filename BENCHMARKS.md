# Benchmarks

Every optimisation in this repo is paired with a before/after number from this file. Numbers
below were measured in a **2-vCPU Intel Xeon @ 2.80 GHz container (AVX2, Ubuntu 24.04, gcc 13.3,
`-O3 -march=x86-64-v2`)** — a small, noisy machine. Re-run on your own box and replace them; the
*ratios* are what matter, and they should hold or improve on a desktop with more cores.

## Workload

- **CI workload (synthetic):** `--synthetic 600 --size 1920x1080 --fake-cost 15000` — 600
  generated 1080p frames with 4 moving objects, synthetic colour-threshold detector that spins
  15 ms per batch to emulate a YOLOv8n CPU model. Exercises every stage, queue and the arena;
  needs no FFmpeg/ONNX, so it runs on every PR.
- **Real workload:** `bench/data/traffic_1080p_30s.mp4` + `models/yolov8n.onnx`, build preset
  `full`. Record your numbers in the second table.

Median of 5 runs, reported by `fastframe --runs 5 --bench --json`.

## End-to-end (CI workload, this container)

| Build / mode | FPS (median of 5) | e2e p50 ms | e2e p99 ms | Δ vs previous | What changed |
| --- | --- | --- | --- | --- | --- |
| `--serial`, scalar preprocess | 57.6 | 18.4 | 18.4 | — | **baseline**: one thread, no queues, no pool, batch 1 |
| pipelined, 1 inference thread, batch 1, scalar | 65.8 | 294.9 | 327.7 | +14 % | 5 stage threads + SPSC queues (queues fill: latency up, throughput only slightly up because inference dominates) |
| pipelined, 2 inference threads, batch 4, scalar | 418.5 | 49.2 | 65.5 | +536 % | inference thread pool + batching: the emulated 15 ms/batch cost is amortised over 4 frames and 2 workers |
| pipelined, 2 threads, batch 4, **AVX2 fused preprocess** | **434.5** | 53.2 | 65.5 | +4 % | fused letterbox+normalise kernel; small e2e gain because preprocess is no longer the bottleneck here (see micro table: 2.1x on the kernel itself) |
| **total** | **7.5x** | | | | |

Things that did not help (kept in the repo, off by default):

- `--pin` (core pinning): no measurable change on 2 vCPUs; re-test on ≥ 8 cores.
- Non-temporal (`_mm256_stream_ps`) stores in the normalise kernel: **2.3x slower** (737 µs vs
  312 µs) because the 4.9 MB output is read again by inference moments later, so bypassing the
  cache only adds a round trip to DRAM. Reverted; left as a note because it is a common "obvious"
  SIMD trick that measurement rejected.

## Micro-benchmarks (`bench_micro`, this container)

| Benchmark | Scalar / baseline | AVX2 / optimised | Gain |
| --- | --- | --- | --- |
| Letterbox resize 1080p→640 (nearest) | 783 µs | 340 µs (gather + LUT) | **2.3x** |
| Normalise 640×640 BGR→CHW float | 268 µs | 240 µs | 1.1x — store-bandwidth bound: 4.9 MB written per call; gcc already SSE-vectorises the scalar loop |
| Full preprocess 1080p (two-pass scalar vs fused AVX2) | 1 151 µs | **558 µs** | **2.1x** — the win comes from fusing (no 1.2 MB intermediate write+read), not from wider lanes |
| SPSC queue hand-off, 1 M items | 262 ms (mutex + vector) | **5.0 ms** (lock-free) | **52x** — 204 M items/s |
| Arena acquire+release, 1 thread | — | 36 ns | |
| Arena acquire+release, 8 threads | — | 186 ns | CAS contention on one cache line; by design only Decode and Encode touch it in the pipeline |
| Tracker, 20 objects, Kalman + Hungarian | — | 9.1 µs / frame | invisible next to any real model |

## Real workload (fill in on your machine)

| Commit | Change | FPS | p99 ms | Δ | Evidence |
| --- | --- | --- | --- | --- | --- |
| | baseline `--serial`, fp32 YOLOv8n | | | — | flame_<sha>.svg |
| | pipelined + batch 4 | | | | |
| | AVX2 fused preprocess | | | | perf: `preprocess` self-time before/after |
| | INT8 model (`scripts/quantize_int8.py`) | | | | + mAP@0.5 fp32 vs int8 on 200 COCO-val images |
| | PGO + LTO (`pgo-gen` → `pgo-use` presets, `release-lto`) | | | | |

Commands:

```bash
cmake --preset full && cmake --build --preset full
./build/full/fastframe --input bench/data/traffic_1080p_30s.mp4 --model models/yolov8n.onnx --serial --runs 5 --bench --json
./build/full/fastframe --input bench/data/traffic_1080p_30s.mp4 --model models/yolov8n.onnx --runs 5 --bench --json
scripts/perf_flamegraph.sh --input bench/data/traffic_1080p_30s.mp4 --model models/yolov8n.onnx --quiet
```

## Noise and the CI gate

`scripts/check_regression.py` fails a PR when median FPS drops > 5 % against
`bench/baselines/<last tag>.json`. Five runs on this container varied by ±3 % run-to-run; GitHub
hosted runners are noisier (±10 % between runs was observed in similar projects), which is why the
gate uses a median, a 5 % threshold, and a 20-second workload rather than a 2-second one. If the
gate flakes, raise `--runs`, never the threshold.

## Safety evidence

- ASan, UBSan (gcc 13) and TSan (clang 18): full test suite + 100-frame end-to-end pass, 0 findings.
  TSan's first run **did** flag a race: the inference histogram was written from pool threads.
  Fixed by returning the batch latency through the `future` and recording it on the stage thread
  (see `InferenceStage::drain_one`). Kept here as the honest "what TSan found" story.
- libFuzzer on `FrameHeaderParser`: 18.2 M executions in 30 s locally, 0 crashes; runs 5 min per
  PR and 30 min nightly through ClusterFuzzLite.
- clang-tidy (`bugprone-*, performance-*, concurrency-*`) clean with `WarningsAsErrors: '*'`.
