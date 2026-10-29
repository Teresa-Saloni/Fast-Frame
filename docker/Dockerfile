# Multi-stage: toolchain + vcpkg build, then a slim runtime image.
#   docker build -f docker/Dockerfile -t fastframe .
#   docker run --rm fastframe --synthetic 300 --size 1920x1080 --bench
# Pin the base image by digest once you have docker: `docker pull ubuntu:24.04` then add @sha256:...
FROM ubuntu:24.04 AS build
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential g++-13 cmake ninja-build git curl ca-certificates pkg-config \
      libgtest-dev libbenchmark-dev \
      libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
    && rm -rf /var/lib/apt/lists/*
# ONNX Runtime prebuilt (CPU). Pin the version; update deliberately.
ARG ORT_VERSION=1.22.0
RUN curl -fsSL https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz \
    | tar -xz -C /opt && mv /opt/onnxruntime-linux-x64-${ORT_VERSION} /opt/onnxruntime
WORKDIR /src
COPY . .
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=x86-64-v2" \
      -DFF_WITH_FFMPEG=ON -DFF_WITH_ORT=ON -DONNXRUNTIME_ROOT=/opt/onnxruntime -DFF_BUILD_BENCH=OFF \
    && cmake --build build -j && ctest --test-dir build --output-on-failure

# Pin the base image by digest once you have docker: `docker pull ubuntu:24.04` then add @sha256:...
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 libavcodec60 libavformat60 libswscale7 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/onnxruntime/lib/libonnxruntime.so* /usr/local/lib/
COPY --from=build /src/build/fastframe /usr/local/bin/fastframe
COPY models/ /models/
ENV LD_LIBRARY_PATH=/usr/local/lib
ENTRYPOINT ["fastframe"]
CMD ["--help"]
