# Interview notes — what each file lets you claim

| File | Claim you can make | The number / proof |
| --- | --- | --- |
| `spsc_queue.hpp` | I implemented a wait-free SPSC ring buffer with correct acquire/release ordering and no false sharing | 204 M items/s vs 5 M/s for a mutex queue; TSan clean; `alignas(64)` test |
| `arena.hpp/.cpp` | Zero malloc on the hot path; lock-free free-list with an ABA tag | 36 ns acquire+release; 8-thread no-double-handout test; `page-faults` flat in perf stat |
| `preprocess_avx2.cpp` | I wrote AVX2 intrinsics after profiling, kept a scalar reference, proved bit-exactness, and measured — including an optimisation that failed | 2.1x fused kernel; streaming stores 2.3x *slower* and reverted |
| `pipeline.cpp` | Pipeline parallelism with bounded queues and explicit back-pressure; ordered output despite a parallel inference pool | 7.5x over serial; `frames_out == frames_in`, in-order test |
| `thread_pool.hpp` | I know when a mutex is the right tool | batches are 15–30 ms; lock cost is noise |
| `tracker.cpp` | Kalman filter + Hungarian assignment from scratch | crossing-objects test keeps ids; 9 µs/frame for 20 objects |
| `frame_header.cpp` + `fuzz/` | I fuzz the code I own and never trust a length field | 18 M iterations, 0 crashes; "lying size field" test |
| `histogram.hpp` | Allocation-free log-linear latency histogram, single-writer per thread | TSan found the one place it was multi-writer; fixed by moving the record to the stage thread |
| `.github/workflows/ci.yml` | A PR cannot merge if it is slower than the last release | median-of-5, 5 % gate, baseline per tag |
| `CMakeLists.txt` | Only one TU gets `-mavx2`; runtime dispatch; sanitizer/hardened/PGO presets | binary runs on non-AVX2 CPUs |

Answer shape for every question: **what I built → why that choice over the alternative → the number that proves it.**
