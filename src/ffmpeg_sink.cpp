// FFmpeg encode backend: BGR24 -> YUV420P (libswscale) -> H.264 (libx264 via
// libavcodec) -> mp4 (libavformat).
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

#include <stdexcept>
#include <string>

#include "fastframe/io.hpp"

namespace ff {

namespace {

class FfmpegSink final : public FrameSink {
 public:
  FfmpegSink(const std::string& path, int w, int h, int fps) : w_(w), h_(h) {
    avformat_alloc_output_context2(&fmt_, nullptr, nullptr, path.c_str());
    if (fmt_ == nullptr) throw std::runtime_error("cannot create output " + path);
    const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (codec == nullptr) throw std::runtime_error("no H.264 encoder (build FFmpeg with x264)");
    stream_ = avformat_new_stream(fmt_, codec);
    ctx_ = avcodec_alloc_context3(codec);
    ctx_->width = w; ctx_->height = h;
    ctx_->time_base = AVRational{1, fps};
    ctx_->framerate = AVRational{fps, 1};
    ctx_->pix_fmt = AV_PIX_FMT_YUV420P;
    ctx_->gop_size = fps;
    av_opt_set(ctx_->priv_data, "preset", "veryfast", 0);
    av_opt_set(ctx_->priv_data, "crf", "23", 0);
    if (fmt_->oformat->flags & AVFMT_GLOBALHEADER) ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    if (avcodec_open2(ctx_, codec, nullptr) < 0) throw std::runtime_error("cannot open encoder");
    avcodec_parameters_from_context(stream_->codecpar, ctx_);
    stream_->time_base = ctx_->time_base;
    if (!(fmt_->oformat->flags & AVFMT_NOFILE) && avio_open(&fmt_->pb, path.c_str(), AVIO_FLAG_WRITE) < 0)
      throw std::runtime_error("cannot open file for writing");
    if (avformat_write_header(fmt_, nullptr) < 0) throw std::runtime_error("cannot write header");
    frm_ = av_frame_alloc();
    frm_->format = AV_PIX_FMT_YUV420P; frm_->width = w; frm_->height = h;
    av_frame_get_buffer(frm_, 32);
    pkt_ = av_packet_alloc();
    sws_ = sws_getContext(w, h, AV_PIX_FMT_BGR24, w, h, AV_PIX_FMT_YUV420P, SWS_BILINEAR, nullptr, nullptr, nullptr);
  }
  ~FfmpegSink() override {
    sws_freeContext(sws_);
    av_packet_free(&pkt_);
    av_frame_free(&frm_);
    avcodec_free_context(&ctx_);
    if (fmt_ != nullptr) {
      if (!(fmt_->oformat->flags & AVFMT_NOFILE)) avio_closep(&fmt_->pb);
      avformat_free_context(fmt_);
    }
  }

  void write(std::span<const std::uint8_t> bgr, int /*w*/, int /*h*/, std::int64_t /*pts*/, std::span<const Track>) override {
    const std::uint8_t* src[1] = {bgr.data()};
    int stride[1] = {w_ * 3};
    av_frame_make_writable(frm_);
    sws_scale(sws_, src, stride, 0, h_, frm_->data, frm_->linesize);
    frm_->pts = next_pts_++;
    avcodec_send_frame(ctx_, frm_);
    drain();
  }

  void finish() override {
    avcodec_send_frame(ctx_, nullptr);
    drain();
    av_write_trailer(fmt_);
  }

 private:
  void drain() {
    while (avcodec_receive_packet(ctx_, pkt_) == 0) {
      av_packet_rescale_ts(pkt_, ctx_->time_base, stream_->time_base);
      pkt_->stream_index = stream_->index;
      av_interleaved_write_frame(fmt_, pkt_);
      av_packet_unref(pkt_);
    }
  }

  AVFormatContext* fmt_ = nullptr;
  AVStream* stream_ = nullptr;
  AVCodecContext* ctx_ = nullptr;
  AVFrame* frm_ = nullptr;
  AVPacket* pkt_ = nullptr;
  SwsContext* sws_ = nullptr;
  int w_, h_;
  std::int64_t next_pts_ = 0;
};

}  // namespace

std::unique_ptr<FrameSink> open_ffmpeg_sink(const std::string& path, int w, int h, int fps) {
  return std::make_unique<FfmpegSink>(path, w, h, fps);
}

}  // namespace ff
