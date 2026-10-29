# Getting started (your 6-week plan, mapped to this repo)

The code is complete; the work left is **measuring on real hardware, with a real model**, and
writing the numbers down. That is the part interviewers ask about.

## Week 1 — build, baseline, commit
1. `git init && git add . && git commit -m "feat: FastFrame v0 — pipeline, arena, SPSC, AVX2, tests, CI"`
2. Push to GitHub; CI must be green on the first push (it needs no external deps).
3. Install FFmpeg dev packages + ONNX Runtime tarball; `python3 scripts/export_yolo.py`.
4. Download any 30 s 1080p clip to `bench/data/traffic_1080p_30s.mp4`.
5. Run `--serial --runs 5 --bench --json` → first row of the real-workload table in BENCHMARKS.md.
6. `git tag v0.1.0` and run `scripts/make_baseline.py bench/results.json v0.1.0`.

## Week 2 — pipeline numbers
- `--inference-threads 1 --batch 1`, then defaults, then `--batch 8`, `--inference-threads 4`. One row each.
- `scripts/perf_flamegraph.sh` on the serial and the pipelined build; commit both SVGs to `docs/`.

## Week 3 — profiling story
- Read the flame graph. Expect `sws_scale` (FFmpeg colour conversion) and ORT kernels on top; the
  AVX2 preprocess should be small. If `sws_scale` dominates, try `SWS_POINT` or request BGR from
  the decoder — that is a legitimate optimisation row.
- `perf stat -e cache-misses,page-faults` with `--arena 8` vs `--arena 64`: show page faults are flat.

## Week 4 — model-side optimisation
- `scripts/quantize_int8.py`; run with `--model models/yolov8n_int8.onnx`; compute mAP@0.5 on a
  200-image COCO-val slice for both (Ultralytics `model.val()` on the ONNX files is enough).
- PGO: `cmake --preset pgo-gen`, run the benchmark, `cmake --preset pgo-use`, run again. One row.

## Week 5 — hardening evidence
- Run every sanitizer preset on the `full` build with the real clip (100 frames is enough).
- Leave the fuzzer running overnight: `./build/fuzz/fuzz/fuzz_parser fuzz/corpus -max_total_time=28800`.
- Turn on ClusterFuzzLite (needs the `fuzz.yml` workflow + a repo with Actions enabled).

## Week 6 — ship
- README headline numbers from YOUR machine; replace the container numbers.
- `git tag v1.0.0` → release.yml records the baseline and pushes the Docker image to GHCR.
- Rehearse the 35-second pitch and the 15 interview answers (see the methodology doc).
