#include "fastframe/pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <deque>
#include <future>
#include <sstream>
#include <vector>

#include "fastframe/preprocess.hpp"

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace ff {

// ----------------------------------------------------------------- Stage bits that need a TU

void Stage::pin_to_cpu(int cpu) noexcept {
#if defined(__linux__)
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(static_cast<unsigned>(cpu), &set);
  (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
  (void)cpu;
#endif
}

// ----------------------------------------------------------------- Decode

class Pipeline::DecodeStage final : public Stage {
 public:
  DecodeStage(Pipeline& p) : Stage("decode"), p_(p) {}
  std::uint64_t frames_in = 0, dropped = 0;

 protected:
  void process(FrameHandle&) override {}
  void run(const std::stop_token& st) override {
    auto& arena = *p_.arena_;
    const auto w = p_.src_->width(), h = p_.src_->height();
    const std::size_t bytes = static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3;
    std::uint32_t seq = 0;
    Backoff bo;
    while (!st.stop_requested()) {
      std::optional<std::uint32_t> slot = arena.acquire();
      while (!slot) {
        if (st.stop_requested()) return;
        if (p_.cfg_.drop_when_full) {
          // Drop policy: pull and discard the frame so the source advances.
          std::int64_t pts;
          std::vector<std::uint8_t>& junk = scratch(bytes);
          if (!p_.src_->next(junk, pts)) { (void)emit(eos_handle(), st); return; }
          ++frames_in; ++dropped;
        } else {
          bo.pause();
        }
        slot = arena.acquire();
      }
      bo.reset();
      FrameHandle fh{*slot, seq, 0};
      {
        ScopedTimer t(hist());
        p_.t_in_[*slot] = std::chrono::steady_clock::now();
        auto* dst = reinterpret_cast<std::uint8_t*>(arena.data(*slot));
        if (!p_.src_->next(std::span<std::uint8_t>(dst, bytes), fh.pts)) {
          arena.release(*slot);
          (void)emit(eos_handle(), st);
          return;
        }
        auto& m = arena.meta(*slot);
        m.width = w; m.height = h; m.pts = fh.pts; m.seq = seq;
        m.detections.clear(); m.tracks.clear();
      }
      ++frames_in; ++seq;
      if (!emit(fh, st)) return;
    }
  }

 private:
  static std::vector<std::uint8_t>& scratch(std::size_t n) {
    static thread_local std::vector<std::uint8_t> s;
    if (s.size() < n) s.resize(n);
    return s;
  }
  Pipeline& p_;
};

// ----------------------------------------------------------------- Preprocess

class Pipeline::PreprocessStage final : public Stage {
 public:
  explicit PreprocessStage(Pipeline& p) : Stage("preprocess"), p_(p) {
    const auto m = static_cast<std::size_t>(p_.cfg_.model_size);
    scratch_.resize(m * m * 3);
  }

 protected:
  void process(FrameHandle& h) override {
    auto& arena = *p_.arena_;
    const auto& m = arena.meta(h.index);
    const std::size_t src_bytes = static_cast<std::size_t>(m.width) * static_cast<std::size_t>(m.height) * 3;
    const auto model = static_cast<std::size_t>(p_.cfg_.model_size);
    Letterbox lb;
    preprocess_frame(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(arena.data(h.index)), src_bytes),
                     m.width, m.height, p_.cfg_.model_size, scratch_,
                     std::span<float>(arena.model_input(h.index), 3 * model * model), lb);
    letterboxes_[h.index] = lb;
  }

 public:
  std::vector<Letterbox>& letterboxes() { return letterboxes_; }

 private:
  Pipeline& p_;
  std::vector<std::uint8_t> scratch_;
  std::vector<Letterbox> letterboxes_;  // sized to arena capacity by Pipeline
};

// ----------------------------------------------------------------- Inference

class Pipeline::InferenceStage final : public Stage {
 public:
  InferenceStage(Pipeline& p, std::vector<Letterbox>& lbs) : Stage("inference"), p_(p), lbs_(lbs) {}
  /// Serial baseline: run one single-frame batch on the calling thread.
  void infer_inline(FrameHandle h) { hist().record_us(infer_batch(std::vector<FrameHandle>{h})); }

 protected:
  void process(FrameHandle&) override {}

  // Batches up to cfg.batch handles (or waits batch_wait_ms), runs each batch
  // on the thread pool, and emits in submission order so downstream stays ordered.
  void run(const std::stop_token& st) override {
    using Clock = std::chrono::steady_clock;
    struct Inflight { std::vector<FrameHandle> handles; std::future<std::uint64_t> done; };
    std::deque<Inflight> inflight;
    std::vector<FrameHandle> batch;
    Backoff bo;
    FrameHandle h;
    auto batch_start = Clock::now();

    auto flush = [&] {
      if (batch.empty()) return;
      std::vector<FrameHandle> hs = batch;
      batch.clear();
      inflight.push_back(Inflight{hs, p_.pool_->submit([this, hs] { return infer_batch(hs); })});
      batch_start = Clock::now();
    };
    auto drain_one = [&]() -> bool {
      auto& f = inflight.front();
      // The batch ran on a pool thread; its latency is recorded HERE, on the
      // stage thread, so the histogram stays single-writer (TSan-clean).
      hist().record_us(f.done.get());
      for (const auto& fh : f.handles) {
        if (!emit(fh, st)) return false;
      }
      inflight.pop_front();
      return true;
    };

    while (!st.stop_requested()) {
      if (in()->try_pop(h)) {
        bo.reset();
        if (is_eos(h)) break;
        if (batch.empty()) batch_start = Clock::now();
        batch.push_back(h);
        if (static_cast<int>(batch.size()) >= p_.cfg_.batch) flush();
      } else {
        if (!batch.empty() && Clock::now() - batch_start >= std::chrono::milliseconds(p_.cfg_.batch_wait_ms)) flush();
        bo.pause();
      }
      // Keep at most pool.size() batches in flight; emit the oldest when full.
      while (inflight.size() >= p_.pool_->size()) {
        if (!drain_one()) return;
      }
    }
    flush();
    while (!inflight.empty()) { if (!drain_one()) return; }
    (void)emit(eos_handle(), st);
  }

 private:
  // Runs on a pool thread. Returns the batch latency in microseconds.
  std::uint64_t infer_batch(const std::vector<FrameHandle>& hs) {
    const auto t0 = std::chrono::steady_clock::now();
    auto& arena = *p_.arena_;
    std::vector<const float*> inputs;
    std::vector<std::vector<Detection>> outs(hs.size());
    inputs.reserve(hs.size());
    for (const auto& fh : hs) inputs.push_back(arena.model_input(fh.index));
    p_.det_->infer(inputs, outs, p_.cfg_.conf_thresh, p_.cfg_.nms_iou);
    // Map model-space boxes back to source pixels through the letterbox.
    for (std::size_t i = 0; i < hs.size(); ++i) {
      const auto& lb = lbs_[hs[i].index];
      auto& m = arena.meta(hs[i].index);
      m.detections.clear();
      for (auto d : outs[i]) {
        d.x1 = (d.x1 - static_cast<float>(lb.pad_x)) / lb.scale;
        d.x2 = (d.x2 - static_cast<float>(lb.pad_x)) / lb.scale;
        d.y1 = (d.y1 - static_cast<float>(lb.pad_y)) / lb.scale;
        d.y2 = (d.y2 - static_cast<float>(lb.pad_y)) / lb.scale;
        d.x1 = std::clamp(d.x1, 0.0f, static_cast<float>(m.width));
        d.x2 = std::clamp(d.x2, 0.0f, static_cast<float>(m.width));
        d.y1 = std::clamp(d.y1, 0.0f, static_cast<float>(m.height));
        d.y2 = std::clamp(d.y2, 0.0f, static_cast<float>(m.height));
        if (d.x2 > d.x1 && d.y2 > d.y1) m.detections.push_back(d);
      }
    }
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
  }

  Pipeline& p_;
  std::vector<Letterbox>& lbs_;
};

// ----------------------------------------------------------------- Tracking

class Pipeline::TrackingStage final : public Stage {
 public:
  explicit TrackingStage(Pipeline& p)
      : Stage("tracking"), p_(p),
        tracker_(Tracker::Params{p.cfg_.track_max_age, p.cfg_.track_min_hits, p.cfg_.track_iou}) {}

 protected:
  void process(FrameHandle& h) override {
    auto& m = p_.arena_->meta(h.index);
    m.tracks = tracker_.update(m.detections);
  }

 private:
  Pipeline& p_;
  Tracker tracker_;
};

// ----------------------------------------------------------------- Encode

class Pipeline::EncodeStage final : public Stage {
 public:
  explicit EncodeStage(Pipeline& p) : Stage("encode"), p_(p) {}
  std::uint64_t frames_out = 0;

 protected:
  void process(FrameHandle& h) override {
    auto& arena = *p_.arena_;
    auto& m = arena.meta(h.index);
    const std::size_t bytes = static_cast<std::size_t>(m.width) * static_cast<std::size_t>(m.height) * 3;
    auto* px = reinterpret_cast<std::uint8_t*>(arena.data(h.index));
    draw_tracks(std::span<std::uint8_t>(px, bytes), m.width, m.height, m.tracks);
    p_.sink_->write(std::span<const std::uint8_t>(px, bytes), m.width, m.height, m.pts, m.tracks);
    ++frames_out;
    const auto dt = std::chrono::steady_clock::now() - p_.t_in_[h.index];
    p_.e2e_.record_us(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(dt).count()));
    arena.release(h.index);  // frame returns to the pool here and only here
  }
  void on_eos() override { p_.sink_->finish(); }

 private:
  Pipeline& p_;
};

// ----------------------------------------------------------------- Pipeline

Pipeline::Pipeline(PipelineConfig cfg, std::unique_ptr<FrameSource> src, std::unique_ptr<Detector> det,
                   std::unique_ptr<FrameSink> sink)
    : cfg_(cfg), src_(std::move(src)), det_(std::move(det)), sink_(std::move(sink)) {
  cfg_.width = src_->width();
  cfg_.height = src_->height();
  cfg_.model_size = det_->input_size();
  const std::size_t frame_bytes = static_cast<std::size_t>(cfg_.width) * static_cast<std::size_t>(cfg_.height) * 3;
  const std::size_t model_bytes = std::size_t{3} * static_cast<std::size_t>(cfg_.model_size) * static_cast<std::size_t>(cfg_.model_size) * sizeof(float);
  arena_ = std::make_unique<FrameArena>(frame_bytes, cfg_.arena_frames, model_bytes);
  pool_ = std::make_unique<ThreadPool>(static_cast<std::size_t>(std::max(1, cfg_.inference_threads)));
  t_in_.resize(cfg_.arena_frames);

  for (int i = 0; i < 4; ++i) queues_.push_back(std::make_unique<FrameQueue>());

  auto dec = std::make_unique<DecodeStage>(*this);
  auto pre = std::make_unique<PreprocessStage>(*this);
  pre->letterboxes().resize(cfg_.arena_frames);
  auto inf = std::make_unique<InferenceStage>(*this, pre->letterboxes());
  auto trk = std::make_unique<TrackingStage>(*this);
  auto enc = std::make_unique<EncodeStage>(*this);

  dec->connect(nullptr, queues_[0].get());
  pre->connect(queues_[0].get(), queues_[1].get());
  inf->connect(queues_[1].get(), queues_[2].get());
  trk->connect(queues_[2].get(), queues_[3].get());
  enc->connect(queues_[3].get(), nullptr);

  stages_.push_back(std::move(dec));
  stages_.push_back(std::move(pre));
  stages_.push_back(std::move(inf));
  stages_.push_back(std::move(trk));
  stages_.push_back(std::move(enc));
}

Pipeline::~Pipeline() {
  for (auto& s : stages_) s->request_stop();
  for (auto& s : stages_) s->join();
}

void Pipeline::run_serial() {
  // The "before" number: one thread, no queues, no pool, no batching. Same
  // arena and kernels so the comparison isolates the pipeline architecture.
  auto* dec = static_cast<DecodeStage*>(stages_[0].get());
  auto* pre = static_cast<PreprocessStage*>(stages_[1].get());
  auto* inf = static_cast<InferenceStage*>(stages_[2].get());
  auto* trk = static_cast<TrackingStage*>(stages_[3].get());
  auto* enc = static_cast<EncodeStage*>(stages_[4].get());
  const auto w = src_->width(), h = src_->height();
  const std::size_t bytes = static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3;
  std::uint32_t seq = 0;
  for (;;) {
    const auto slot = arena_->acquire();
    if (!slot) break;  // cannot happen serially: encode releases every frame
    FrameHandle fh{*slot, seq, 0};
    bool more;
    {
      ScopedTimer t(dec->hist());
      t_in_[*slot] = std::chrono::steady_clock::now();
      auto* dst = reinterpret_cast<std::uint8_t*>(arena_->data(*slot));
      more = src_->next(std::span<std::uint8_t>(dst, bytes), fh.pts);
      if (more) {
        auto& m = arena_->meta(*slot);
        m.width = w; m.height = h; m.pts = fh.pts; m.seq = seq;
        m.detections.clear(); m.tracks.clear();
      }
    }
    if (!more) { arena_->release(*slot); break; }
    ++dec->frames_in; ++seq;
    pre->process_inline(fh);
    inf->infer_inline(fh);
    trk->process_inline(fh);
    enc->process_inline(fh);
  }
  sink_->finish();
}

PipelineResult Pipeline::run() {
  const auto t0 = std::chrono::steady_clock::now();
  if (cfg_.serial) {
    run_serial();
  } else {
    int cpu = 0;
    for (auto& s : stages_) s->start(cfg_.pin_threads ? cpu++ : -1);
    for (auto& s : stages_) s->join();
  }
  const auto t1 = std::chrono::steady_clock::now();

  PipelineResult r;
  auto* dec = static_cast<DecodeStage*>(stages_[0].get());
  auto* enc = static_cast<EncodeStage*>(stages_[4].get());
  r.frames_in = dec->frames_in;
  r.frames_dropped = dec->dropped;
  r.frames_out = enc->frames_out;
  r.arena_exhausted = arena_->exhausted_count();
  r.wall_seconds = std::chrono::duration<double>(t1 - t0).count();
  r.fps = r.wall_seconds > 0 ? static_cast<double>(r.frames_out) / r.wall_seconds : 0.0;
  r.e2e_p50_us = e2e_.percentile_us(0.50);
  r.e2e_p99_us = e2e_.percentile_us(0.99);
  r.backend_preprocess = std::string(preprocess_backend());
  r.backend_detector = det_->name();
  for (auto& s : stages_) {
    const auto& h = s->histogram();
    r.stages.push_back(StageStats{std::string(s->name()), h.count(), h.mean_us(), h.percentile_us(0.5),
                                  h.percentile_us(0.99), h.max_us(), s->stall_cycles()});
  }
  return r;
}

std::string PipelineResult::to_json() const {
  std::ostringstream o;
  o << "{\"fps\":" << fps << ",\"frames\":" << frames_out << ",\"frames_in\":" << frames_in
    << ",\"dropped\":" << frames_dropped << ",\"arena_exhausted\":" << arena_exhausted
    << ",\"wall_s\":" << wall_seconds << ",\"p50_ms\":" << static_cast<double>(e2e_p50_us) / 1000.0
    << ",\"p99_ms\":" << static_cast<double>(e2e_p99_us) / 1000.0
    << ",\"preprocess\":\"" << backend_preprocess << "\",\"detector\":\"" << backend_detector << "\",\"stages\":[";
  for (std::size_t i = 0; i < stages.size(); ++i) {
    const auto& s = stages[i];
    o << (i ? "," : "") << "{\"name\":\"" << s.name << "\",\"count\":" << s.frames << ",\"mean_us\":" << s.mean_us
      << ",\"p50_us\":" << s.p50_us << ",\"p99_us\":" << s.p99_us << ",\"max_us\":" << s.max_us
      << ",\"stalls\":" << s.stalls << "}";
  }
  o << "]}";
  return o.str();
}

std::string PipelineResult::to_table() const {
  std::ostringstream o;
  o << "frames " << frames_out << "/" << frames_in << "  dropped " << frames_dropped << "  arena-exhausted "
    << arena_exhausted << "\n"
    << "wall " << wall_seconds << " s   fps " << fps << "   e2e p50 " << static_cast<double>(e2e_p50_us) / 1000.0
    << " ms  p99 " << static_cast<double>(e2e_p99_us) / 1000.0 << " ms\n"
    << "preprocess=" << backend_preprocess << " detector=" << backend_detector << "\n"
    << "stage        count    mean_us    p50_us    p99_us    max_us   stalls\n";
  for (const auto& s : stages) {
    o.width(12); o << std::left << s.name << std::right;
    o.width(6); o << s.frames;
    o.width(11); o << static_cast<std::uint64_t>(s.mean_us);
    o.width(10); o << s.p50_us;
    o.width(10); o << s.p99_us;
    o.width(10); o << s.max_us;
    o.width(9); o << s.stalls << "\n";
  }
  return o.str();
}

}  // namespace ff
