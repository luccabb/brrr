# syntax=docker/dockerfile:1
#
# build:   compiles brrr with the CUDA toolkit
# export:  just the binary (docker buildx build --target export --output type=local,dest=out .)
# runtime: the image people run, with its own cuBLASLt and cuDNN; needs only the NVIDIA
#          container runtime

ARG CUDA_VERSION=13.4.2

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build
ARG ARCHS="100 103"
ARG CUDNN_VERSION=9.27.0.42-1
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
      libcudnn9-cuda-13=${CUDNN_VERSION} libcudnn9-dev-cuda-13=${CUDNN_VERSION} \
      libcudnn9-headers-cuda-13=${CUDNN_VERSION} \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
# cuDNN's runtime-compiled kernels (MXFP4) need NVRTC next to it, and cudnn-frontend opens
# the shared CUDA runtime.
RUN make -j"$(nproc)" ARCHS="${ARCHS}" build/brrr && ./build/brrr --help >/dev/null \
 && mkdir -p build/lib \
 && cp -L /usr/local/cuda/lib64/libcublasLt.so.13 /usr/local/cuda/lib64/libcudart.so.13 \
          /usr/local/cuda/lib64/libnvrtc.so.13 \
          /usr/local/cuda/lib64/libnvrtc-builtins.so.13.* /usr/lib/*-linux-gnu/libcudnn.so.9 \
          /usr/lib/*-linux-gnu/libcudnn_graph.so.9 /usr/lib/*-linux-gnu/libcudnn_heuristic.so.9 \
          /usr/lib/*-linux-gnu/libcudnn_engines_*.so.9 build/lib/

FROM scratch AS export
COPY --from=build /src/build/brrr /brrr
COPY LICENSE NOTICE /
COPY --from=build /src/build/cudnn-frontend/LICENSE /LICENSE.cudnn-frontend

FROM ubuntu:24.04 AS runtime
COPY --from=build /src/build/brrr /usr/local/bin/brrr
COPY --from=build /src/build/lib/ /usr/local/lib/
RUN ldconfig
# The driver caches what it prepares from cuBLASLt (~400 MB on GB300) under $HOME.
ENV CUDA_CACHE_PATH=/tmp/.nv/ComputeCache
COPY LICENSE NOTICE /usr/share/doc/brrr/
COPY --from=build /src/build/cudnn-frontend/LICENSE /usr/share/doc/brrr/LICENSE.cudnn-frontend
WORKDIR /tmp
ENTRYPOINT ["/usr/local/bin/brrr"]
