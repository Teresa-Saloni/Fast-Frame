// FFmpeg decode backend: demux (libavformat) -> decode (libavcodec, frame
// threading on) -> convert to BGR24 (libswscale) straight into the arena slot.
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <cstring>
#include <stdexcept>
#include <string>

#include "fastframe/io.hpp"

namespace ff {

namespace {

class FfmpegSource final : public FrameSource {
 public:
  explicit FfmpegSource(const std::string& path) {
    if (avformat_open_input(&fmt_, path.c_str(), nullptr, nullptr) < 0) throw std::runtime_error("cannot open " + path);
    if (avformat_find_stream_info(fmt_, nullptr) < 0) throw std::runtime_error("no stream info");
    const AVCodec* codec = nullptr;
    stream_ = av_find_best_stream(fmt_, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
    if (stream_ < 0 || codec == nullptr) throw std::runtime_error("no video stream");
    ctx_ = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(ctx_, fmt_->streams[stream_]->codecpar);
    ctx_->thread_count = 0;                 // let libavcodec pick
    ctx_->thread_type = FF_THREAD_FRAME;    // frame threading = decode parallelism inside the stage
    if (avcodec_open2(ctx_, codec, nullptr) < 0) throw std::runtime_error("cannot open codec");
    w_ = ctx_->width; h_ = ctx_->height;
    pkt_ = av_packet_alloc();
    frm_ = av_frame_alloc();
  }
  ~FfmpegSource() override {
    av_frame_free(&frm_);
    av_packet_free(&pkt_);
    sws_freeContext(sws_);
    avcodec_free_context(&ctx_);
    avformat_close_input(&fmt_);
  }

  int width() const noexcept override { return w_; }
  int height() const noexcept override { return h_; }

  bool next(std::span<std::uint8_t> dst, std::int64_t& pts) override {
    for (;;) {
      int r = avcodec_receive_frame(ctx_, frm_);
      if (r == 0) {
        if (sws_ == nullptr) {
          sws_ = sws_getContext(w_, h_, static_cast<AVPixelFormat>(frm_->format), w_, h_, AV_PIX_FMT_BGR24,
                                SWS_BILINEAR, nullptr, nullptr, nullptr);
        }
        std::uint8_t* planes[1] = {dst.data()};
        int strides[1] = {w_ * 3};
        sws_scale(sws_, frm_->data, frm_->linesize, 0, h_, planes, strides);
        pts = frm_->pts == AV_NOPTS_VALUE ? frm_->best_effort_timestamp : frm_->pts;
        av_frame_unref(frm_);
        return true;
      }
      if (r == AVERROR_EOF) return false;
      if (r != AVERROR(EAGAIN)) throw std::runtime_error("decode error");
      // Need more input.
      if (eof_sent_) continue;
      r = av_read_frame(fmt_, pkt_);
      if (r < 0) {
        avcodec_send_packet(ctx_, nullptr);  // flush
        eof_sent_ = true;
        continue;
      }
      if (pkt_->stream_index == stream_) avcodec_send_packet(ctx_, pkt_);
      av_packet_unref(pkt_);
    }
  }

 private:
  AVFormatContext* fmt_ = nullptr;
  AVCodecContext* ctx_ = nullptr;
  SwsContext* sws_ = nullptr;
  AVPacket* pkt_ = nullptr;
  AVFrame* frm_ = nullptr;
  int stream_ = -1, w_ = 0, h_ = 0;
  bool eof_sent_ = false;
};

}  // namespace

std::unique_ptr<FrameSource> open_ffmpeg_source(const std::string& path) { return std::make_unique<FfmpegSource>(path); }

}  // namespace ff
