// fastframe CLI
//
//   fastframe --input video.mp4 --model yolov8n.onnx --output out.mp4
//   fastframe --synthetic 900 --bench --json            # no deps needed
//   fastframe --input video.mp4 --model m.onnx --bench --runs 5 --json
//
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "fastframe/io.hpp"
#include "fastframe/pipeline.hpp"
#include "fastframe/preprocess.hpp"

namespace {

struct Args {
  std::string input, output, model;
  int synthetic = 0;
  int width = 1920, height = 1080;
  int runs = 1;
  bool bench = false, json = false, quiet = false;
  int fake_cost_us = 0;
  ff::PipelineConfig cfg;
};

void usage() {
  std::puts(
      "fastframe " FF_VERSION "\n"
      "  --input PATH         input video (needs FF_WITH_FFMPEG)\n"
      "  --output PATH        output mp4 with boxes (needs FF_WITH_FFMPEG)\n"
      "  --model PATH         YOLO ONNX model (needs FF_WITH_ORT); else synthetic detector\n"
      "  --synthetic N        use a generated N-frame source instead of --input\n"
      "  --size WxH           synthetic frame size (default 1920x1080)\n"
      "  --fake-cost US       synthetic detector spin time per batch (emulates a model)\n"
      "  --bench              print throughput/latency; --runs K repeats and reports the median\n"
      "  --json               machine-readable output (used by CI)\n"
      "  --batch N  --inference-threads N  --arena N  --batch-wait-ms N\n"
      "  --pin                pin stage threads to cores 0..4\n"
      "  --serial             baseline mode: all stages on one thread, no queues/pool/batching\n"
      "  --drop               drop frames when the arena is exhausted instead of blocking\n"
      "  --conf F --nms F --max-age N --min-hits N\n");
}

bool parse(int argc, char** argv, Args& a) {
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
    if (k == "--help" || k == "-h") { usage(); return false; }
    else if (k == "--input") a.input = next() ? argv[i] : "";
    else if (k == "--output") a.output = next() ? argv[i] : "";
    else if (k == "--model") a.model = next() ? argv[i] : "";
    else if (k == "--synthetic") a.synthetic = std::atoi(next() ? argv[i] : "0");
    else if (k == "--size") { if (next()) std::sscanf(argv[i], "%dx%d", &a.width, &a.height); }
    else if (k == "--fake-cost") a.fake_cost_us = std::atoi(next() ? argv[i] : "0");
    else if (k == "--bench") a.bench = true;
    else if (k == "--json") a.json = true;
    else if (k == "--quiet") a.quiet = true;
    else if (k == "--runs") a.runs = std::max(1, std::atoi(next() ? argv[i] : "1"));
    else if (k == "--batch") a.cfg.batch = std::max(1, std::atoi(next() ? argv[i] : "4"));
    else if (k == "--batch-wait-ms") a.cfg.batch_wait_ms = std::atoi(next() ? argv[i] : "8");
    else if (k == "--inference-threads") a.cfg.inference_threads = std::max(1, std::atoi(next() ? argv[i] : "2"));
    else if (k == "--arena") a.cfg.arena_frames = static_cast<std::uint32_t>(std::max(8, std::atoi(next() ? argv[i] : "64")));
    else if (k == "--pin") a.cfg.pin_threads = true;
    else if (k == "--serial") a.cfg.serial = true;
    else if (k == "--drop") a.cfg.drop_when_full = true;
    else if (k == "--conf") a.cfg.conf_thresh = static_cast<float>(std::atof(next() ? argv[i] : "0.25"));
    else if (k == "--nms") a.cfg.nms_iou = static_cast<float>(std::atof(next() ? argv[i] : "0.45"));
    else if (k == "--max-age") a.cfg.track_max_age = std::atoi(next() ? argv[i] : "30");
    else if (k == "--min-hits") a.cfg.track_min_hits = std::atoi(next() ? argv[i] : "3");
    else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); usage(); return false; }
  }
  if (a.input.empty() && a.synthetic == 0) { usage(); return false; }
  return true;
}

std::unique_ptr<ff::FrameSource> make_source(const Args& a) {
  if (a.synthetic > 0) return std::make_unique<ff::SyntheticSource>(a.width, a.height, static_cast<std::uint32_t>(a.synthetic));
#if defined(FF_WITH_FFMPEG)
  return ff::open_ffmpeg_source(a.input);
#else
  std::fprintf(stderr, "built without FFmpeg: use --synthetic N or rebuild with -DFF_WITH_FFMPEG=ON\n");
  return nullptr;
#endif
}

std::unique_ptr<ff::Detector> make_detector(const Args& a) {
  if (!a.model.empty()) {
#if defined(FF_WITH_ORT)
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    return ff::open_ort_detector(a.model, std::max(1, hw - 4));
#else
    std::fprintf(stderr, "built without ONNX Runtime: rebuild with -DFF_WITH_ORT=ON\n");
    return nullptr;
#endif
  }
  return std::make_unique<ff::SyntheticDetector>(640, a.fake_cost_us);
}

std::unique_ptr<ff::FrameSink> make_sink(const Args& a, int w, int h) {
  if (!a.output.empty()) {
#if defined(FF_WITH_FFMPEG)
    return ff::open_ffmpeg_sink(a.output, w, h, 30);
#else
    (void)w; (void)h;
    std::fprintf(stderr, "built without FFmpeg: --output ignored\n");
#endif
  }
  return std::make_unique<ff::NullSink>();
}

}  // namespace

int main(int argc, char** argv) {
  Args a;
  if (!parse(argc, argv, a)) return 2;

  std::vector<ff::PipelineResult> results;
  for (int run = 0; run < a.runs; ++run) {
    auto src = make_source(a);
    auto det = make_detector(a);
    if (!src || !det) return 1;
    auto sink = make_sink(a, src->width(), src->height());
    ff::Pipeline p(a.cfg, std::move(src), std::move(det), std::move(sink));
    results.push_back(p.run());
    if (!a.quiet && !a.json && a.runs > 1) std::fprintf(stderr, "run %d: %.2f fps\n", run + 1, results.back().fps);
  }

  // Median by fps (what the CI gate reads).
  std::sort(results.begin(), results.end(), [](const auto& x, const auto& y) { return x.fps < y.fps; });
  const auto& med = results[results.size() / 2];
  if (a.json) {
    std::printf("%s\n", med.to_json().c_str());
  } else if (a.bench || !a.quiet) {
    std::printf("%s", med.to_table().c_str());
  }
  return med.frames_out == med.frames_in - med.frames_dropped ? 0 : 1;
}
