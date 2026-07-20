# API reference

Public API of `spconv_triton.pytorch`, the drop-in namespace for `spconv.pytorch`. Import it once and alias it.

```python
import spconv_triton.pytorch as spconv
```

Every symbol below matches spconv 2.x in name, signature, and behavior unless noted. Runtime toggles and environment variables are documented in [ENVIRONMENT_VARIABLES.md](ENVIRONMENT_VARIABLES.md).

## Coordinate conventions

Read this before building a network. Getting axis order wrong produces silently wrong results, not errors.

`SparseConvTensor.indices` has shape `[N, ndim + 1]`. Column 0 is the batch index. The remaining columns are spatial coordinates in the **same order** as `spatial_shape`.

The convolution, pooling, transpose, inverse, and low-level ops are **agnostic** to whether spatial coordinates are `xyz` or `zyx`. They apply `kernel_size`, `stride`, `padding`, and `dilation` per axis in `spatial_shape` order, so a consistent order is all they need — but that consistency is yours to enforce.

Only voxelization fixes an order, and it fixes exactly one: it consumes an XYZ point cloud and emits ZYX voxel `indices`. The whole XYZ→ZYX flip happens there and nowhere else.

| Item | Order |
|---|---|
| Constructor `vsize_xyz`, `coors_range_xyz` | XYZ |
| Point cloud input `pc[:, :ndim]` | XYZ |
| Output voxel `indices` | ZYX (reversed) |
| Public attrs `vsize`, `grid_size`, `grid_stride`, `coors_range` | ZYX (reversed) |
| `num_per_voxel`, `pc_voxel_id` | axis-agnostic |

> [!IMPORTANT]
> The voxelizer returns ZYX `indices`, so build `SparseConvTensor` with `spatial_shape` in that **same ZYX order**, and the network runs ZYX end to end. A `spatial_shape` whose axis order disagrees with `indices` is the classic footgun: the layers accept it and produce silently wrong results, never an error.

## SparseConvTensor

The sparse tensor container. Carries features, integer coordinates, and cached indice pairs.

```python
spconv.SparseConvTensor(features, indices, spatial_shape, batch_size, grid=None, voxel_num=None, indice_dict=None, benchmark=False, permanent_thrust_allocator=False, enable_timer=False, force_algo=None)
```

| Arg | Description |
|---|---|
| `features` | float tensor `[N, C]` |
| `indices` | int32 tensor `[N, ndim + 1]`, batch index in column 0, spatial columns in `spatial_shape` axis order |
| `spatial_shape` | spatial grid size, list or array of length `ndim`, same axis order as the spatial columns of `indices` (see [coordinate conventions](#coordinate-conventions)) |
| `batch_size` | number of samples in the batch |
| `grid` | optional preallocated grid buffer for repeated pair generation |
| `voxel_num`, `indice_dict`, `benchmark`, `force_algo` | optional, rarely set by hand |

| Member | Description |
|---|---|
| `.features` | feature tensor `[N, C]`, read-only, assign via `replace_feature` |
| `.indices` | coordinate tensor `[N, ndim + 1]` |
| `.spatial_shape` | spatial grid size |
| `.batch_size` | batch count |
| `replace_feature(feat)` | return a new tensor with features replaced, fx and autograd safe |
| `dense(channels_first=True)` | densify to `[batch, C, *spatial_shape]`, or `[batch, *spatial_shape, C]` when `False` |
| `from_dense(x)` | classmethod, build from a channel-last dense tensor |
| `find_indice_pair(key)` | cached indice-pair data for an `indice_key`, or `None` |
| `shadow_copy()` | shallow copy sharing all members |

Assigning to `.features` raises `ValueError`. Use `replace_feature`. `select_by_index` also raises, matching spconv.

## Convolution layers

Submanifold convolution keeps the input coordinates fixed. Regular sparse convolution generates new active sites under stride and padding.

```python
spconv.SubMConv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, indice_key=None, algo=None, fp32_accum=None, large_kernel_fast_algo=False, name=None)
```

| Arg | Description |
|---|---|
| `in_channels`, `out_channels` | feature dimensions |
| `kernel_size` | int or per-axis sequence, submanifold requires odd sizes |
| `stride`, `padding`, `dilation` | int or per-axis sequence |
| `groups` | must be 1, grouped convolution is not supported |
| `bias` | add a learnable bias |
| `indice_key` | name under which indice pairs are cached and reused |
| `algo` | `ConvAlgo`, defaults to the upstream heuristic |
| `fp32_accum` | `True` forces fp32 accumulation, `False` follows the fp16 policy, `None` follows the global flag |

`SubMConv{1,2,3,4}d` use this signature. `SparseConv{1,2,3,4}d` and the transposed variants add `record_voxel_count=False`, which tracks active voxel counts for `get_max_num_voxels`. Activation parameters exist only on the internal base class and are not part of the public leaf constructors.

Reuse one `indice_key` across layers that share a geometry so pair generation runs once. Submanifold layers on the same resolution should share a key. A downsampling `SparseConv` and its matching `SparseInverseConv` share a key so the inverse recovers the exact input sites.

### Transposed and inverse convolution

Transposed convolution learns an upsampling. Inverse convolution reverses a specific earlier downsampling and reuses its cached pairs, so it takes an `indice_key` and no stride or padding.

```python
spconv.SparseConvTranspose3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, indice_key=None, algo=None, fp32_accum=None, record_voxel_count=False, large_kernel_fast_algo=False, name=None)
```

```python
spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key, bias=True, algo=None, fp32_accum=None, large_kernel_fast_algo=False, name=None)
```

`SparseConvTranspose{1,2,3,4}d` and `SparseInverseConv{1,2,3,4}d` follow these two shapes. `SparseConvTranspose` has no `output_padding` argument, matching spconv.

## Pooling

```python
spconv.SparseMaxPool3d(kernel_size, stride=None, padding=0, dilation=1, indice_key=None, algo=None, record_voxel_count=False, name=None)
```

| Class | Notes |
|---|---|
| `SparseMaxPool{1,2,3,4}d` | max over each window, accumulator starts at 0 |
| `SparseAvgPool{1,2,3}d` | mean over real active point count, no 4d variant |
| `SparseGlobalMaxPool` | global max, constructor takes only `name` |
| `SparseGlobalAvgPool` | global mean, returns shape `[batch]` matching an upstream quirk |

`SparseGlobalMaxPool` and `SparseGlobalAvgPool` take no pooling arguments.

## Containers and activations

| Symbol | Purpose |
|---|---|
| `SparseSequential(*modules)` | sequential container that dispatches sparse and dense modules |
| `SparseModule` | base class marking a module sparse-aware |
| `SparseReLU`, `SparseBatchNorm`, `SparseIdentity` | fx and quantization safe activation and norm shims |
| `ToDense` | densify a `SparseConvTensor` inside a `SparseSequential` |
| `RemoveGrid` | drop the cached grid buffer |
| `assign_name_for_sparse_modules(module)` | assign stable names for indice-key sharing |

Dense `nn.Module` layers such as `BatchNorm1d` and `ReLU` acting on `.features` may be placed directly in a `SparseSequential`.

## Tables

Merge modules combine parallel branches on aligned or misaligned coordinates.

| Symbol | Purpose |
|---|---|
| `AddTable` | elementwise add of sparse tensors sharing coordinates |
| `ConcatTable` | concatenate features of a shared branch |
| `JoinTable` | join a list of sparse tensors |

## Voxelization

Convert a raw point cloud into voxels and coordinates. Construct after selecting the device. Read the [coordinate conventions](#coordinate-conventions) first.

```python
from spconv_triton.utils import Point2VoxelGPU3d

voxelizer = Point2VoxelGPU3d(vsize_xyz, coors_range_xyz, num_point_features, max_num_voxels, max_num_points_per_voxel)
```

| Arg | Description |
|---|---|
| `vsize_xyz` | voxel size per axis in XYZ |
| `coors_range_xyz` | `[xmin, ymin, zmin, xmax, ymax, zmax]` |
| `num_point_features` | feature count per point, including coordinates |
| `max_num_voxels` | cap on generated voxels |
| `max_num_points_per_voxel` | cap on points aggregated per voxel |

> [!IMPORTANT]
> The constructor takes XYZ (`vsize_xyz`, `coors_range_xyz`), but the public attributes `vsize`, `grid_size`, `grid_stride`, and `coors_range` read back **ZYX-reversed** to match spconv. Never feed a read-back attribute into a fresh constructor: the silent axis swap corrupts your grid. Keep your own XYZ values for reconstruction.

`Point2VoxelGPU{1,2,3,4}d` run on the accelerator, `Point2VoxelCPU{1,2,3,4}d` on the host. The device-generic `PointToVoxel` in `spconv_triton.pytorch.utils` takes an extra `device` argument instead of fixing the backend.

Voxelization is the object call itself. Calling the voxelizer invokes `__call__`, and `generate_voxel_with_id` is the same call that additionally returns the per-point voxel id.

```python
voxels, indices, num_per_voxel = voxelizer(pc, clear_voxels=True, empty_mean=False)
voxels, indices, num_per_voxel, pc_voxel_id = voxelizer.generate_voxel_with_id(pc, clear_voxels=True, empty_mean=False)
```

| Arg | Description |
|---|---|
| `pc` | point cloud `[N, num_point_features]`, first `ndim` columns are XYZ coordinates |
| `clear_voxels` | zero the reused voxel buffer before filling |
| `empty_mean` | average each voxel over its real point count instead of leaving zero-padded slots |

| Return | Shape | Description |
|---|---|---|
| `voxels` | `[num_voxels, max_num_points_per_voxel, num_point_features]` | per-voxel point features, zero-padded |
| `indices` | `[num_voxels, ndim]` | voxel coordinates, int32, ZYX |
| `num_per_voxel` | `[num_voxels]` | points per voxel, clamped at `max_num_points_per_voxel` |
| `pc_voxel_id` | `[N]` | voxel id per input point, int64, `-1` for dropped points |

Points fill voxels first-come-first-served. `gather_features_by_pc_voxel_id(features, pc_voxel_id)` from `spconv_triton.pytorch.utils` scatters per-voxel results back to the original points.

## Ops and functional

`spconv_triton.pytorch.ops` and `spconv_triton.pytorch.functional` expose the low-level pair generation, indice convolution, implicit-GEMM, and pooling primitives that the layers build on. Most models never call these directly. Key entry points are `get_indice_pairs`, `get_indice_pairs_implicit_gemm`, `implicit_gemm`, and the autograd `Function` wrappers in `functional`. Pair-generation ops assume batch-major linearization over `spatial_shape` and apply kernel parameters per axis in that order.

`functional.sparse_add(*tensors)` adds sparse tensors that share a shape but hold different active sites. `spconv_triton.pytorch.hash.HashTable(device, key_dtype, value_dtype, max_size=-1)` is the device-agnostic hash table used for coordinate lookups.

## Enums, constants, version

| Symbol | Values |
|---|---|
| `ConvAlgo` | `Native`, `MaskImplicitGemm`, `MaskSplitImplicitGemm` |
| `AlgoHint` | `NoHint`, `Fowrard`, `BackwardInput`, `BackwardWeight` |

`ConvAlgo` selects the pair and GEMM strategy. The default follows spconv, `MaskImplicitGemm` when `kernel volume <= 32` else `Native`. `MaskSplitImplicitGemm` is never chosen automatically. There is no runtime override beyond the per-layer `algo` argument.

`spconv_triton.constants.SPCONV_ALLOW_TF32` is a module constant, default `False`, read live by the fp32 GEMM. Set it through the top-level `spconv_triton` package, since the aliased `spconv.constants` resolves to a version-only module.

`spconv_triton.__version__` reports the package version. `SPCONV_VERSION_NUMBERS` reports the spconv version targeted for parity.
