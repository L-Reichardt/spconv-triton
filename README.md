![spconv-Triton](assets/github_banner.png)

# spconv-Triton

Hardware-agnostic sparse convolution for PyTorch. Submanifold and regular sparse convolution in 1D to 4D, transposed and inverse convolution, and pooling, written in [Triton](https://github.com/triton-lang/triton) so the same code runs on NVIDIA, AMD, and any other Triton-capable accelerator.

spconv-Triton is a drop-in replacement for [spconv 2.x](https://github.com/traveller59/spconv), the sparse convolution backbone behind LiDAR 3D object detection and segmentation stacks such as OpenPCDet and mmdetection3d. Change one import and existing models, checkpoints, and training loops keep working.

<div align="center">

✅ Verified with: NVIDIA RTX 3060, A100, H100, L4, B200 &nbsp;•&nbsp; AMD MI300X &nbsp;•&nbsp; `torch 2.4 – 2.13` &nbsp;•&nbsp; `python 3.10 – 3.14`

</div>

## Why spconv-Triton

Sparse convolution is the backbone of 3D perception and generation on point clouds and voxels. [spconv](https://github.com/traveller59/spconv) is the fast, widely adopted implementation, but it is no longer maintained and its prebuilt CUDA kernels tie you to NVIDIA hardware and a narrow band of CUDA, PyTorch, and Python versions.

spconv-Triton reimplements the same operators in Triton. You get the full operator set on current PyTorch and Python across vendors, with numerical parity to spconv.

Reach for it when spconv does not fit your stack, whether that means AMD hardware, a newer PyTorch or Python, or a deployment where only Triton is available. spconv-Triton keeps the spconv API, so nothing else in your model changes.

## Installation

Install the PyTorch build for your accelerator first. It brings the matching Triton flavor.

Then install the package.

```bash
pip install spconv-triton
```

> [!NOTE]  
> spconv-Triton does not declare Triton as a dependency.
> PyTorch already ships the correct Triton for your backend, e.g. `triton` on CUDA and `pytorch-triton-rocm` on ROCm. Declaring it would pull the CUDA wheel onto ROCm installs and break them.
> With [uv](https://docs.astral.sh/uv/), `UV_TORCH_BACKEND=auto` resolves the right PyTorch automatically.

## Quickstart

Full documentation lives in the [API reference](docs/API.md). Refer to it when building your model.

If adopting a network originating from spconv, change the import. Everything else stays the same.

```python
import torch
# import spconv.pytorch as spconv        # before
import spconv_triton.pytorch as spconv   # after

features = torch.randn(1000, 3, device="cuda")
indices = torch.zeros(1000, 4, dtype=torch.int32, device="cuda")   # column 0 is batch
indices[:, 1:] = torch.randint(0, 64, (1000, 3), device="cuda")    # spatial coords, zyx

x = spconv.SparseConvTensor(features, indices, spatial_shape=[64, 64, 64], batch_size=1)

net = spconv.SparseSequential(
    spconv.SubMConv3d(3, 32, 3, padding=1, indice_key="subm0"),
    spconv.SparseConv3d(32, 64, 3, stride=2, padding=1, indice_key="down0"),
    spconv.SparseInverseConv3d(64, 32, 3, indice_key="down0"),
).cuda()

out = net(x)          # SparseConvTensor
dense = out.dense()   # [batch, C, *spatial_shape]
```

`SparseConvTensor` carries `features` of shape `[N, C]` and integer `indices` of shape `[N, ndim + 1]` with the batch index in column 0. The remaining index columns and `spatial_shape` must use the same axis order. Convolution and pooling layers are agnostic to whether that order is `xyz` or `zyx`.
The voxelization helpers take `xyz` point coordinates and ranges and return `zyx` indices, following spconv, so a voxelized pipeline runs in `zyx`.

## Training

### TF32

TF32 is off by default, matching spconv. It speeds up the fp32 path on supported hardware. Enable it through the top-level package.

```python
import spconv_triton
spconv_triton.constants.SPCONV_ALLOW_TF32 = True
```

Set it on `spconv_triton`, not the aliased `spconv`. The alias resolves to a version-only module, so assigning there sets a dead attribute and silently leaves TF32 off. On some AMD hardware with torch 2.7+, TF32 also needs `HIPBLASLT_ALLOW_TF32=1` in the environment.

### Mixed precision

Sparse convolution inherits spconv's autocast behavior, which differs from dense `nn.Conv2d`. Under `torch.autocast` the layers run fp16 in training mode but stay fp32 in eval mode. For fp16 inference, cast the model explicitly with `.half()`.

fp16 GEMMs accumulate in fp32 by default for cross-hardware accuracy. Set `SPCONV_ALLOW_FP16_ACCUM=1` to accumulate in fp16 like spconv, which helps only on consumer Ampere and Ada. Avoid this for large batch training.

### Compile cache

Triton compiles and autotunes kernels on first use. Point `TRITON_CACHE_DIR` at a stable path, default `~/.triton/cache`, so compiled kernels and, on Triton 3.3.0+, the autotune results persist across processes. For distributed training, warm the cache with a short single-process run before the first multi-rank launch so ranks do not autotune in parallel.

> [!TIP]
> Use Trition 3.3.0+ to avoid recompiling every run.

### torch.compile

The sparse layers run eagerly and let `torch.compile` fuse the dense parts of the model around them.

```python
model = torch.compile(model, fullgraph=False)   # fuses BatchNorm, ReLU, and residual adds on .features
```

Sparse convolution is data-dependent. Point counts, neighborhoods, and sometimes grid sizes change with every input, and the kernels are already written fused, so `torch.compile` treats them as opaque calls. Use `fullgraph=False`.

## Migrating from spconv

Models, `state_dict` checkpoints, and training code transfer without changes or retraining. Only the import changes. A checkpoint trained with spconv in OpenPCDet or mmdetection3d loads and runs unchanged.

Two behaviors stay faithful to spconv rather than PyTorch conventions, both covered under [Training](#training). Autocast keeps eval-mode output in fp32, and TF32 stays off until you enable it.

Box operations such as NMS are not ported, since most frameworks provide their own. CPU-only inference and training are not supported, as the compute kernels are GPU-only by design.

## Performance

Parity is verified against spconv 2.3.8 by a frozen golden data test suite.

On NVIDIA, relative speed depends on precision. fp16 and TF32 are competitive to faster, most so on server GPUs.
fp32 is slower, across the board.
spconv does not run on AMD at all. On MI300X, measured against FlexGEMM on submanifold convolution, the one common operator, spconv-Triton is faster and additionally covers the full operator set.

Benchmarks report warm-cache medians over 10 runs of 200 warmup iterations and 1000 measured iterations each, at a representative width and voxel count (C256, 50k voxels), against spconv, FlexGEMM, warpconvnet, and fVDB where each is available.

| Hardware | Plots |
|---|---|
| NVIDIA RTX 3060 (Ampere) | [docs/RTX3060](docs/RTX3060/) |
| NVIDIA A100 (Ampere) | [docs/A100](docs/A100/) |
| NVIDIA H100 (Hopper) | [docs/H100](docs/H100/) |
| NVIDIA L4 (Lovelace) | [docs/L4](docs/L4/) |
| AMD MI300X (submanifold) | [docs/MI300X](docs/MI300X/) |

NVIDIA T4 (Turing) runs correctly but slowly. Prefer spconv there.

## Documentation

| Topic | Location |
|---|---|
| API reference | [docs/API.md](docs/API.md) |
| Environment variables | [docs/ENVIRONMENT_VARIABLES.md](docs/ENVIRONMENT_VARIABLES.md) |
| Changelog | [CHANGELOG.md](CHANGELOG.md) |

## Testing

A GPU is required. The frozen suite runs against committed golden data and never needs reference spconv.

```bash
uv run pytest tests/ -q -p no:cacheprovider
```

The `tox` matrix covers the support corners across Python, PyTorch, and both CUDA and ROCm runtimes. See [tox.ini](tox.ini) for the full env list.

## Attribution

Derived from [spconv](https://github.com/traveller59/spconv) by Yan Yan and contributors, originally licensed under Apache License 2.0. This work is likewise released under Apache License 2.0.
