# brrr

low precision GPU stress test.

brrr is heavily inspired by [gpu-burn](https://github.com/wilicc/gpu-burn), extended to lower precision formats: BF16, FP8, MXFP8, MXFP4 and NVFP4. brrr fills 90% (`-m`) of each GPU's memory and checks every result bit for bit while burning the GPUs, it uses cuBLASLt and cuDNN.

**Requirements:** 
- NVIDIA driver 580 or newer
- CUDA 13's `libcublasLt.so.13`
- for MXFP4: cuDNN 9 for CUDA 13, `libcudart.so.13` and `libnvrtc.so.13`

## Building

```
git clone https://github.com/luccabb/brrr
cd brrr
docker build -t brrr .
docker run --rm --gpus all brrr 60
```

You can pass build arguments to specify the CUDA version, the compute capability and the cuDNN version, e.g.:

```
docker build \
  --build-arg CUDA_VERSION=13.4.2 \
  --build-arg ARCHS=103 \
  --build-arg CUDNN_VERSION=9.27.0.42-1 \
  -t brrr .
```

Or with CUDA 13 installed:

```
make              # build brrr
make ARCHS=100    # compute capability to build for
```

On Kubernetes:

```yaml
apiVersion: v1
kind: Pod
metadata: {name: brrr}
spec:
  restartPolicy: Never
  containers:
  - name: brrr
    image: ghcr.io/luccabb/brrr:main
    args: ["60"]
    resources: {limits: {nvidia.com/gpu: 8}}
```

## Quick start

Burns BF16 on all GPUs for 60 seconds:

```
$ brrr 60
GPU 0: NVIDIA GB300 (compute 10.3)
GPU 1: NVIDIA GB300 (compute 10.3)
GPU 2: NVIDIA GB300 (compute 10.3)
GPU 3: NVIDIA GB300 (compute 10.3)
cuBLASLt 13.8.0: /usr/local/cuda/lib64/libcublasLt.so.13
Burning for 60 seconds: bf16
GPU 3: using 247.9 GiB of 275.9 GiB free (276.6 GiB total)
GPU 2: using 247.9 GiB of 275.9 GiB free (276.6 GiB total)
GPU 0: using 247.9 GiB of 275.9 GiB free (276.6 GiB total)
GPU 1: using 247.9 GiB of 275.9 GiB free (276.6 GiB total)
 10.0%  gpu0 bf16 errors:0 | gpu1 bf16 errors:0 | gpu2 bf16 errors:0 | gpu3 bf16 errors:0 |
 20.0%  gpu0 bf16 1725 TF 62C errors:0 | gpu1 bf16 1769 TF 62C errors:0 | gpu2 bf16 1757 TF 60C errors:0 | gpu3 bf16 1765 TF 61C errors:0 |
 30.0%  gpu0 bf16 1722 TF 62C errors:0 | gpu1 bf16 1768 TF 63C errors:0 | gpu2 bf16 1756 TF 61C errors:0 | gpu3 bf16 1764 TF 61C errors:0 |
 40.0%  gpu0 bf16 1720 TF 62C errors:0 | gpu1 bf16 1766 TF 63C errors:0 | gpu2 bf16 1754 TF 61C errors:0 | gpu3 bf16 1762 TF 61C errors:0 |
 50.0%  gpu0 bf16 1719 TF 62C errors:0 | gpu1 bf16 1766 TF 63C errors:0 | gpu2 bf16 1753 TF 61C errors:0 | gpu3 bf16 1761 TF 62C errors:0 |
 60.0%  gpu0 bf16 1718 TF 63C errors:0 | gpu1 bf16 1765 TF 63C errors:0 | gpu2 bf16 1752 TF 61C errors:0 | gpu3 bf16 1761 TF 62C errors:0 |
 70.0%  gpu0 bf16 1717 TF 63C errors:0 | gpu1 bf16 1764 TF 63C errors:0 | gpu2 bf16 1752 TF 61C errors:0 | gpu3 bf16 1760 TF 62C errors:0 |
 80.0%  gpu0 bf16 1717 TF 63C errors:0 | gpu1 bf16 1764 TF 63C errors:0 | gpu2 bf16 1751 TF 61C errors:0 | gpu3 bf16 1760 TF 62C errors:0 |
 90.0%  gpu0 bf16 1717 TF 63C errors:0 | gpu1 bf16 1763 TF 63C errors:0 | gpu2 bf16 1751 TF 61C errors:0 | gpu3 bf16 1759 TF 62C errors:0 |
100.0%  gpu0 bf16 1716 TF 63C errors:0 | gpu1 bf16 1763 TF 63C errors:0 | gpu2 bf16 1750 TF 61C errors:0 | gpu3 bf16 1759 TF 61C errors:0 |
done

gpu  format          TFLOPS  vs peers  errors  maxT   maxW  verdict
0    bf16            1716.5     -2.2%       0   63C   1375  OK
1    bf16            1762.8     +0.5%       0   63C   1378  OK
2    bf16            1750.4     -0.2%       0   61C   1392  OK
3    bf16            1758.7     +0.2%       0   62C   1395  OK

Tested 4 GPUs:
	GPU 0: OK
	GPU 1: OK
	GPU 2: OK
	GPU 3: OK
```

```
$ brrr --help
Usage: brrr [options] [seconds]       (default 10 s, split across formats)
  -f, --formats LIST     comma-separated formats, or all (default: bf16)
  -i, --gpu INDEX        one GPU (default: all)
  -m N|N%                GPU memory to use: N MiB or N% of free (default 90%)
  -l, --list             list GPUs and the formats each can run
      --json             machine-readable result on stdout
      --per-format       seconds apply to each format, not the total
  -d                     fp64 (gpu-burn compatible)
  -tc                    tf32 (gpu-burn's tensor-core flag, which ran FP16)
  -h, --help
Exit: 0 OK, 1 slow, 2 faulty, 3 could not run.
```

## Formats

Formats are defined in [`src/format.cpp`](src/format.cpp), each with the part of the GPU it exercises. To see all formats:

```
$ brrr -l -i 0
cuBLASLt 13.8.0: /usr/local/lib/libcublasLt.so.13
cuDNN 9.27.0: /usr/local/lib/libcudnn.so.9
GPU 0: NVIDIA GB300 (compute 10.3, 283254 MiB, 0008:06:00.0)
  bf16           supported
  fp16           supported
  tf32           supported
  fp32           supported
  fp64           supported
  fp8            supported
  fp8-out        supported
  mxfp8          supported
  mxfp4          supported (cuDNN)
  nvfp4          supported
```
