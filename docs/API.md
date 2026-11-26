# API reference

This page describes the public API of `spconv_triton.pytorch`. This namespace is a drop-in replacement for `spconv.pytorch`. Import it once and give it an alias.

```python
import spconv_triton.pytorch as spconv
```

Each symbol below matches spconv 2.x in name, signature, and behavior. Exceptions are noted. Runtime switches and environment variables are described in [ENVIRONMENT_VARIABLES.md](ENVIRONMENT_VARIABLES.md).

## Coordinate conventions

Read this section before you build a network. A wrong axis order gives wrong results. It does not raise an error.

`SparseConvTensor.indices` has shape `[N, ndim + 1]`. Column 0 is the batch index. The other columns are spatial coordinates. They use the **same order** as `spatial_shape`.

The convolution, pooling, transpose, inverse, and low-level operators work with either `xyz` or `zyx` order. They apply `kernel_size`, `stride`, `padding`, and `dilation` to each axis in `spatial_shape` order. You must keep the order consistent.

Only voxelization sets an axis order. It takes an XYZ point cloud and returns ZYX voxel `indices`. The XYZ to ZYX change happens only in voxelization.

| Item | Order |
|---|---|
| Constructor `vsize_xyz`, `coors_range_xyz` | XYZ |
| Point cloud input `pc[:, :ndim]` | XYZ |
| Output voxel `indices` | ZYX (reversed) |
| Public attributes `vsize`, `grid_size`, `grid_stride`, `coors_range` | ZYX (reversed) |
| `num_per_voxel`, `pc_voxel_id` | axis-agnostic |

> [!IMPORTANT]
> The voxelizer returns ZYX `indices`. Set `spatial_shape` in the **same ZYX order** when you build `SparseConvTensor`. Then the network runs in ZYX order from start to end. If the axis order of `spatial_shape` differs from `indices`, the layers still run. They return wrong results without an error.

## SparseConvTensor

The sparse tensor container. It stores features, integer coordinates, and cached indice pairs.

```python
spconv.SparseConvTensor(features, indices, spatial_shape, batch_size, grid=None, voxel_num=None, indice_dict=None, benchmark=False, permanent_thrust_allocator=False, enable_timer=False, force_algo=None)
```

| Argument | Description |
|---|---|
| `features` | float tensor of shape `[N, C]` |
| `indices` | int32 tensor of shape `[N, ndim + 1]`. Column 0 is the batch index. The spatial columns use the axis order of `spatial_shape`. See [coordinate conventions](#coordinate-conventions). |
| `spatial_shape` | spatial grid size. A list or array of length `ndim`. Its axis order must match the spatial columns of `indices`. |
| `batch_size` | number of samples in the batch |
| `grid` | optional preallocated grid buffer. Use it for repeated pair generation. |
| `voxel_num`, `indice_dict`, `benchmark`, `force_algo` | optional. You rarely set these by hand. |

| Member | Description |
|---|---|
| `.features` | feature tensor of shape `[N, C]`. Read-only. Use `replace_feature` to assign. |
| `.indices` | coordinate tensor of shape `[N, ndim + 1]` |
| `.spatial_shape` | spatial grid size |
| `.batch_size` | number of samples in the batch |
| `replace_feature(feat)` | returns a new tensor with replaced features. It is safe for FX tracing and autograd. |
| `dense(channels_first=True)` | converts to a dense tensor of shape `[batch, C, *spatial_shape]`. If `channels_first` is `False`, the shape is `[batch, *spatial_shape, C]`. |
| `from_dense(x)` | class method. Builds the sparse tensor from a dense tensor with channels last. |
| `find_indice_pair(key)` | returns the cached indice-pair data for an `indice_key`, or `None` |
| `shadow_copy()` | returns a shallow copy that shares all members |

Assigning to `.features` raises `ValueError`. Use `replace_feature` instead. `select_by_index` also raises an error. This matches spconv.

## Convolution layers

A submanifold convolution keeps the input coordinates fixed. A regular sparse convolution creates new active sites. The stride and padding control where they appear.

```python
spconv.SubMConv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, indice_key=None, algo=None, fp32_accum=None, large_kernel_fast_algo=False, name=None)
```

| Argument | Description |
|---|---|
| `in_channels`, `out_channels` | feature dimensions |
| `kernel_size` | an int, or a sequence with one value per axis. A submanifold convolution requires odd sizes. |
| `stride`, `padding`, `dilation` | an int, or a sequence with one value per axis |
| `groups` | must be 1. Grouped convolution is not supported. |
| `bias` | adds a learnable bias |
| `indice_key` | name under which indice pairs are cached and reused |
| `algo` | a `ConvAlgo`. The default is the upstream heuristic. |
| `fp32_accum` | `True` forces fp32 accumulation. `False` follows the fp16 policy. `None` follows the global flag. |

The classes `SubMConv{1,2,3,4}d` use this signature. The classes `SparseConv{1,2,3,4}d` and the transposed variants also accept `record_voxel_count=False`. When it is `True`, they track the active voxel count for `get_max_num_voxels`. Activation parameters exist only on the internal base class. They are not part of the public constructors.

Reuse one `indice_key` for layers that share a geometry. Then pair generation runs once. Submanifold layers at the same resolution should share a key. A downsampling `SparseConv` and its matching `SparseInverseConv` should share a key. Then the inverse layer recovers the exact input sites.

### Transposed and inverse convolution

A transposed convolution learns an upsampling. An inverse convolution reverses a specific earlier downsampling. It reuses the cached pairs of that layer. So it takes an `indice_key`, and it takes no stride or padding.

```python
spconv.SparseConvTranspose3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, indice_key=None, algo=None, fp32_accum=None, record_voxel_count=False, large_kernel_fast_algo=False, name=None)
```

```python
spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key, bias=True, algo=None, fp32_accum=None, large_kernel_fast_algo=False, name=None)
```

The classes `SparseConvTranspose{1,2,3,4}d` and `SparseInverseConv{1,2,3,4}d` use these two signatures. `SparseConvTranspose` has no `output_padding` argument. This matches spconv.

## Pooling

```python
spconv.SparseMaxPool3d(kernel_size, stride=None, padding=0, dilation=1, indice_key=None, algo=None, record_voxel_count=False, name=None)
```

| Class | Notes |
|---|---|
| `SparseMaxPool{1,2,3,4}d` | maximum over each window. The accumulator starts at 0. |
| `SparseAvgPool{1,2,3}d` | mean over the real active point count. There is no 4D variant. |
| `SparseGlobalMaxPool` | global maximum. The constructor takes only `name`. |
| `SparseGlobalAvgPool` | global mean. It returns shape `[batch]`. This matches an upstream quirk. |

`SparseGlobalMaxPool` and `SparseGlobalAvgPool` take no pooling arguments.

## Containers and activations

| Symbol | Purpose |
|---|---|
| `SparseSequential(*modules)` | sequential container. It calls sparse and dense modules. |
| `SparseModule` | base class. It marks a module as sparse-aware. |
| `SparseReLU`, `SparseBatchNorm`, `SparseIdentity` | activation and normalization modules that are safe for FX tracing and quantization |
| `ToDense` | converts a `SparseConvTensor` to dense inside a `SparseSequential` |
| `RemoveGrid` | removes the cached grid buffer |
| `assign_name_for_sparse_modules(module)` | assigns stable names, so that layers can share `indice_key` values |

Dense `nn.Module` layers, such as `BatchNorm1d` and `ReLU`, can go directly in a `SparseSequential`. They act on `.features`.

## Tables

Merge modules combine parallel branches. The branches can have aligned or misaligned coordinates.

| Symbol | Purpose |
|---|---|
| `AddTable` | adds sparse tensors that share coordinates, element by element |
| `ConcatTable` | concatenates the features of a shared branch |
| `JoinTable` | joins a list of sparse tensors |

## Voxelization

Voxelization converts a raw point cloud into voxels and voxel coordinates. Create the voxelizer after you select the device. Read the [coordinate conventions](#coordinate-conventions) first.

```python
from spconv_triton.utils import Point2VoxelGPU3d

voxelizer = Point2VoxelGPU3d(vsize_xyz, coors_range_xyz, num_point_features, max_num_voxels, max_num_points_per_voxel)
```

| Argument | Description |
|---|---|
| `vsize_xyz` | voxel size for each axis, in XYZ order |
| `coors_range_xyz` | `[xmin, ymin, zmin, xmax, ymax, zmax]` |
| `num_point_features` | number of features per point, including the coordinates |
| `max_num_voxels` | maximum number of generated voxels |
| `max_num_points_per_voxel` | maximum number of points that one voxel aggregates |

> [!IMPORTANT]
> The constructor takes XYZ values (`vsize_xyz`, `coors_range_xyz`). The public attributes `vsize`, `grid_size`, `grid_stride`, and `coors_range` return ZYX values, as spconv does. Do not pass these attributes to a new constructor. The axis order changes, and the grid becomes wrong. Keep your own XYZ values to rebuild the voxelizer.

`Point2VoxelGPU{1,2,3,4}d` run on the accelerator. `Point2VoxelCPU{1,2,3,4}d` run on the host. The device-independent `PointToVoxel` class in `spconv_triton.pytorch.utils` takes a `device` argument. Use it to select the backend.

Calling the voxelizer runs voxelization. The call uses `__call__`. `generate_voxel_with_id` runs the same operation and also returns the voxel ID of each point.

```python
voxels, indices, num_per_voxel = voxelizer(pc, clear_voxels=True, empty_mean=False)
voxels, indices, num_per_voxel, pc_voxel_id = voxelizer.generate_voxel_with_id(pc, clear_voxels=True, empty_mean=False)
```

| Argument | Description |
|---|---|
| `pc` | point cloud of shape `[N, num_point_features]`. The first `ndim` columns are the XYZ coordinates. |
| `clear_voxels` | sets the reused voxel buffer to zero before filling it |
| `empty_mean` | averages each voxel over its real point count. If `False`, the zero-padded slots stay in the average. |

| Return value | Shape | Description |
|---|---|---|
| `voxels` | `[num_voxels, max_num_points_per_voxel, num_point_features]` | features of the points in each voxel, zero-padded |
| `indices` | `[num_voxels, ndim]` | voxel coordinates. int32, ZYX order. |
| `num_per_voxel` | `[num_voxels]` | number of points in each voxel. Capped at `max_num_points_per_voxel`. |
| `pc_voxel_id` | `[N]` | voxel ID of each input point. int64. `-1` marks a dropped point. |

Points fill voxels in first-come, first-served order. The function `gather_features_by_pc_voxel_id(features, pc_voxel_id)` in `spconv_triton.pytorch.utils` maps per-voxel results back to the original points.

## Ops and functional

The modules `spconv_triton.pytorch.ops` and `spconv_triton.pytorch.functional` provide the low-level primitives. These include pair generation, indice convolution, implicit GEMM, and pooling. The layers use these primitives. Most models do not call them directly. The main entry points are `get_indice_pairs`, `get_indice_pairs_implicit_gemm`, `implicit_gemm`, and the autograd `Function` wrappers in `functional`. Pair generation assumes batch-major linearization over `spatial_shape`. It applies the kernel parameters to each axis in that order.

`functional.sparse_add(*tensors)` adds sparse tensors. The tensors must have the same shape. They can have different active sites. `spconv_triton.pytorch.hash.HashTable(device, key_dtype, value_dtype, max_size=-1)` is the device-independent hash table. It looks up coordinates.

## Enums, constants, and version

| Symbol | Values |
|---|---|
| `ConvAlgo` | `Native`, `MaskImplicitGemm`, `MaskSplitImplicitGemm` |
| `AlgoHint` | `NoHint`, `Fowrard`, `BackwardInput`, `BackwardWeight` |

`ConvAlgo` selects the pair strategy and the GEMM strategy. The default follows spconv. It is `MaskImplicitGemm` when the kernel volume is 32 or less. Otherwise it is `Native`. `MaskSplitImplicitGemm` is never selected automatically. The only override is the per-layer `algo` argument.

`spconv_triton.constants.SPCONV_ALLOW_TF32` is a module constant. Its default is `False`. The fp32 GEMM reads it at each call. Set it through the top-level `spconv_triton` package. The alias `spconv.constants` points to a module that holds only the version.

`spconv_triton.__version__` gives the package version. `SPCONV_VERSION_NUMBERS` gives the spconv version that this package matches.
