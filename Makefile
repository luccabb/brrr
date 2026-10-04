# CUDA from $PATH, then the usual install locations.
CUDA ?= $(or $(patsubst %/bin/nvcc,%,$(shell command -v nvcc 2>/dev/null)), \
             $(firstword $(wildcard /usr/local/cuda /usr)))
NVCC ?= $(CUDA)/bin/nvcc
# Compute capabilities to build for (https://developer.nvidia.com/cuda-gpus).
ARCHS ?= 103

GENCODE := $(foreach a,$(ARCHS),-gencode arch=compute_$(a),code=sm_$(a)) \
           -gencode arch=compute_$(lastword $(ARCHS)),code=compute_$(lastword $(ARCHS))
INCLUDES := -Isrc -Ibuild
NVFLAGS := -std=c++17 -O2 --extended-lambda $(GENCODE) $(INCLUDES) -Xcompiler=-Wall
LDLIBS := -lcublasLt -lculibos -lcudart_static -lpthread -ldl -lrt
LDFLAGS := -Xcompiler=-static-libstdc++,-static-libgcc

all: build/brrr

# With cuDNN's headers, brrr gets a cuDNN backend for MXFP4; libcudnn itself is opened at runtime.
# The header-only cudnn-frontend is downloaded at a pinned version and checksum.
CUDNN_INCLUDE ?= $(firstword $(dir $(wildcard $(CUDA)/include/cudnn.h /usr/include/cudnn.h /usr/include/*/cudnn.h)))
CUDNN_FRONTEND_VERSION := 1.30.0
CUDNN_FRONTEND_SHA256 := 2e6a28661b91c0f43c2b24809750b480a93975bcf949c9715e540a7ab8575b31
ifneq ($(CUDNN_INCLUDE),)
NVFLAGS += -DBRRR_CUDNN -isystem $(CUDNN_INCLUDE) -isystem build/cudnn-frontend/include
build/cudnn.cpp.o: build/cudnn-frontend/LICENSE
endif

# Offline builds can put the tarball at build/cudnn-frontend.tgz first.
build/cudnn-frontend.tgz:
	@mkdir -p build
	curl -fsSL -o $@ https://github.com/NVIDIA/cudnn-frontend/archive/refs/tags/v$(CUDNN_FRONTEND_VERSION).tar.gz

build/cudnn-frontend/LICENSE: build/cudnn-frontend.tgz
	echo "$(CUDNN_FRONTEND_SHA256)  $<" | sha256sum -c -
	@mkdir -p build/cudnn-frontend
	tar -xzf $< -C build/cudnn-frontend --strip-components=1 \
	    cudnn-frontend-$(CUDNN_FRONTEND_VERSION)/include cudnn-frontend-$(CUDNN_FRONTEND_VERSION)/LICENSE.txt
	mv build/cudnn-frontend/LICENSE.txt $@

SOURCES := $(wildcard src/*.cpp src/*.cu)
OBJECTS := $(patsubst src/%,build/%.o,$(SOURCES))


build/%.o: src/%
	@mkdir -p $(dir $@)
	$(NVCC) $(NVFLAGS) -c $< -o $@

build/brrr: $(OBJECTS)
	$(NVCC) $(NVFLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS)

image:
	docker buildx build --target runtime -t brrr .

clean:
	rm -rf build

.PHONY: all image clean
