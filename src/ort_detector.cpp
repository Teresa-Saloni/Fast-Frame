// ONNX Runtime detector for YOLOv8/YOLO11-style models exported with
// `yolo export format=onnx dynamic=True` (output [N, 4+classes, anchors]).
//
// Zero-copy input: the arena's CHW float buffers are wrapped as OrtValues
// through CreateTensor on CPU memory — no memcpy into ORT. Batches > 1 use
// one contiguous staging buffer only when the arena slots are not adjacent.

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "fastframe/io.hpp"
#include "fastframe/tracker.hpp"

namespace ff {

namespace {

class OrtDetector final : public Detector {
 public:
  OrtDetector(const std::string& path, int intra_threads)
      : env_(ORT_LOGGING_LEVEL_WARNING, "fastframe"), mem_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(intra_threads);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    session_ = std::make_unique<Ort::Session>(env_, path.c_str(), so);

    Ort::AllocatorWithDefaultOptions alloc;
    in_name_ = session_->GetInputNameAllocated(0, alloc).get();
    out_name_ = session_->GetOutputNameAllocated(0, alloc).get();
    const auto shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    // [N,3,H,W]; dynamic dims are -1 -> default 640.
    size_ = shape.size() == 4 && shape[2] > 0 ? static_cast<int>(shape[2]) : 640;
  }

  int input_size() const noexcept override { return size_; }
  std::string name() const override { return "onnxruntime"; }

  void infer(std::span<const float* const> inputs, std::span<std::vector<Detection>> outputs,
             float conf_thresh, float nms_iou) override {
    const auto n = static_cast<std::int64_t>(inputs.size());
    const std::size_t per = 3ull * static_cast<std::size_t>(size_) * static_cast<std::size_t>(size_);
    const std::int64_t shape[4] = {n, 3, size_, size_};

    // Zero-copy when the batch is one tensor (or the slots happen to be contiguous).
    bool contiguous = true;
    for (std::size_t i = 1; i < inputs.size(); ++i)
      if (inputs[i] != inputs[i - 1] + per) { contiguous = false; break; }
    float* data;
    if (contiguous) {
      data = const_cast<float*>(inputs[0]);
    } else {
      // infer() runs concurrently on pool threads: staging is per-thread.
      static thread_local std::vector<float> staging;
      staging.resize(per * inputs.size());
      for (std::size_t i = 0; i < inputs.size(); ++i) std::memcpy(staging.data() + i * per, inputs[i], per * sizeof(float));
      data = staging.data();
    }
    Ort::Value in = Ort::Value::CreateTensor<float>(mem_, data, per * inputs.size(), shape, 4);

    const char* in_names[] = {in_name_.c_str()};
    const char* out_names[] = {out_name_.c_str()};
    auto outs = session_->Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
    const auto oshape = outs[0].GetTensorTypeAndShapeInfo().GetShape();  // [N, 4+C, A]
    const float* o = outs[0].GetTensorData<float>();
    const auto rows = static_cast<std::size_t>(oshape[1]);
    const auto anchors = static_cast<std::size_t>(oshape[2]);
    const auto classes = static_cast<std::int32_t>(rows - 4);

    for (std::size_t b = 0; b < inputs.size(); ++b) {
      const float* base = o + b * rows * anchors;
      std::vector<Detection> dets;
      for (std::size_t a = 0; a < anchors; ++a) {
        std::int32_t best = 0;
        float score = 0.0f;
        for (std::int32_t c = 0; c < classes; ++c) {
          const float s = base[(4 + static_cast<std::size_t>(c)) * anchors + a];
          if (s > score) { score = s; best = c; }
        }
        if (score < conf_thresh) continue;
        const float cx = base[0 * anchors + a], cy = base[1 * anchors + a];
        const float w = base[2 * anchors + a], h = base[3 * anchors + a];
        dets.push_back(Detection{cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score, best});
      }
      outputs[b] = nms(std::move(dets), nms_iou);
    }
  }

 private:
  Ort::Env env_;
  Ort::MemoryInfo mem_;
  std::unique_ptr<Ort::Session> session_;
  std::string in_name_, out_name_;
  int size_ = 640;
};

}  // namespace

std::unique_ptr<Detector> open_ort_detector(const std::string& model_path, int intra_threads) {
  return std::make_unique<OrtDetector>(model_path, intra_threads);
}

}  // namespace ff
