"""Shared test helpers: implementation resolution, canonicalization, case runners.

These helpers are part of the frozen test contract. They must work identically for
the reference implementation (``SPCONV_TEST_IMPL=spconv``) and the port
(``SPCONV_TEST_IMPL=spconv_triton``).
"""

import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

DATA_DIR = Path(__file__).parent / "data"
DEVICE = "cuda"


# ---------------------------------------------------------------------------
# Implementation resolution
# ---------------------------------------------------------------------------

def get_impl_root() -> str:
    return os.environ.get("SPCONV_TEST_IMPL", "spconv_triton")


def load_impl() -> SimpleNamespace:
    """Import the implementation under test and expose its submodules."""
    root_name = get_impl_root()
    root = importlib.import_module(root_name)
    pytorch = importlib.import_module(f"{root_name}.pytorch")
    return SimpleNamespace(
        name=root_name,
        root=root,
        pytorch=pytorch,
        core=importlib.import_module(f"{root_name}.core"),
        ops=importlib.import_module(f"{root_name}.pytorch.ops"),
        functional=importlib.import_module(f"{root_name}.pytorch.functional"),
        modules=importlib.import_module(f"{root_name}.pytorch.modules"),
        conv=importlib.import_module(f"{root_name}.pytorch.conv"),
        pool=importlib.import_module(f"{root_name}.pytorch.pool"),
        tables=importlib.import_module(f"{root_name}.pytorch.tables"),
        putils=importlib.import_module(f"{root_name}.pytorch.utils"),
        utils=importlib.import_module(f"{root_name}.utils"),
        hash=importlib.import_module(f"{root_name}.pytorch.hash"),
    )


def resolve_act(impl: SimpleNamespace, name: str):
    """Resolve an activation enum member through the implementation's conv module.

    Both spconv and the port expose ``conv.tv.gemm.Activation``.
    """
    return getattr(impl.conv.tv.gemm.Activation, name)


def resolve_algo(impl: SimpleNamespace, name: Optional[str]):
    if name is None:
        return None
    return getattr(impl.core.ConvAlgo, name)


# ---------------------------------------------------------------------------
# Golden data
# ---------------------------------------------------------------------------

_GOLDEN_CACHE: Dict[str, Any] = {}


def load_golden(name: str):
    if name not in _GOLDEN_CACHE:
        path = DATA_DIR / name
        _GOLDEN_CACHE[name] = torch.load(path, map_location="cpu",
                                         weights_only=False)
    return _GOLDEN_CACHE[name]


def golden_case_ids(name: str) -> List[str]:
    return [c["id"] for c in load_golden(name)["cases"]]


def golden_case(name: str, case_id: str) -> Dict[str, Any]:
    for c in load_golden(name)["cases"]:
        if c["id"] == case_id:
            return c
    raise KeyError(case_id)


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------

def linearize_indices(indices: torch.Tensor,
                      spatial_shape: Sequence[int]) -> torch.Tensor:
    """Linearize [N, ndim+1] (batch, *coords) indices, batch-major, row-major."""
    assert indices.dim() == 2 and indices.shape[1] == len(spatial_shape) + 1
    lin = indices[:, 0].long()
    for i, s in enumerate(spatial_shape):
        lin = lin * int(s) + indices[:, i + 1].long()
    return lin


def canonical_order(indices: torch.Tensor,
                    spatial_shape: Sequence[int]) -> torch.Tensor:
    return torch.argsort(linearize_indices(indices, spatial_shape))


def expected_out_shape(spatial_shape: Sequence[int],
                       args: Dict[str, Any]) -> List[int]:
    """Output spatial shape for pair-gen args (impl-independent formulas)."""
    if args["subm"]:
        return list(spatial_shape)
    out = []
    for i, s in enumerate(spatial_shape):
        k, st = args["ksize"][i], args["stride"][i]
        p, d = args["padding"][i], args["dilation"][i]
        if args["transpose"]:
            out.append((s - 1) * st - 2 * p + k + args["out_padding"][i])
        else:
            out.append(1 if k == -1 else (s + 2 * p - d * (k - 1) - 1) // st + 1)
    return out


def inverse_permutation(order: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=order.device)
    return inv


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def assert_tensor_equal(actual: torch.Tensor, expected: torch.Tensor, msg: str = ""):
    actual = actual.detach().cpu()
    expected = expected.detach().cpu()
    assert actual.shape == tuple(expected.shape) or actual.shape == expected.shape, \
        f"{msg}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    assert actual.dtype == expected.dtype, \
        f"{msg}: dtype {actual.dtype} != {expected.dtype}"
    if not torch.equal(actual, expected):
        diff = (actual != expected)
        n_bad = int(diff.sum())
        raise AssertionError(
            f"{msg}: tensors not exactly equal ({n_bad}/{actual.numel()} mismatched)")


def assert_close(actual: torch.Tensor, expected: torch.Tensor, atol: float,
                 msg: str = ""):
    actual = actual.detach().cpu().to(torch.float64)
    expected = expected.detach().cpu().to(torch.float64)
    assert actual.shape == expected.shape, \
        f"{msg}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    if actual.numel() == 0:
        return
    err = (actual - expected).abs().max().item()
    assert err <= atol, (
        f"{msg}: max abs err {err:.6e} > atol {atol:.6e} "
        f"(ref max {expected.abs().max().item():.4e})")


def assert_sparse_output(out, expect: Dict[str, Any], dtype: torch.dtype,
                         check_features: bool = True, msg: str = ""):
    """Compare a SparseConvTensor against canonicalized golden output."""
    assert list(out.spatial_shape) == list(expect["out_spatial_shape"]), \
        f"{msg}: spatial shape {out.spatial_shape} != {expect['out_spatial_shape']}"
    order = canonical_order(out.indices.cpu(), out.spatial_shape)
    idx_canon = out.indices.cpu()[order]
    assert_tensor_equal(idx_canon, expect["out_indices"], f"{msg}: out indices")
    if check_features:
        assert out.features.dtype == dtype, \
            f"{msg}: feature dtype {out.features.dtype} != {dtype}"
        feats_canon = out.features.detach().cpu()[order]
        assert_close(feats_canon, expect["out_features"], expect["atol_out"],
                     f"{msg}: out features")
    return order


# ---------------------------------------------------------------------------
# Layer / pipeline construction & execution
# ---------------------------------------------------------------------------

def build_layer(impl: SimpleNamespace, spec: Dict[str, Any],
                dtype: torch.dtype, device: str):
    if spec["cls"] == "SparseSequential":
        children = [build_layer(impl, c, dtype, device)
                    for c in spec["children"]]
        return impl.pytorch.SparseSequential(*children)
    if spec["cls"].startswith("nn."):
        cls = getattr(torch.nn, spec["cls"][3:])
    elif hasattr(impl.pytorch, spec["cls"]):
        cls = getattr(impl.pytorch, spec["cls"])
    elif hasattr(impl.conv, spec["cls"]):
        cls = getattr(impl.conv, spec["cls"])
    else:
        cls = getattr(impl.pool, spec["cls"])
    ctor = dict(spec["ctor"])
    if "act_type" in ctor and isinstance(ctor["act_type"], str):
        ctor["act_type"] = resolve_act(impl, ctor["act_type"])
    if "algo" in ctor and isinstance(ctor["algo"], str):
        ctor["algo"] = resolve_algo(impl, ctor["algo"])
    layer = cls(**ctor)
    layer = layer.to(device)
    if dtype == torch.float16:
        layer = layer.half()
    params = spec.get("params") or {}
    with torch.no_grad():
        for name, value in params.items():
            if value is None:
                continue
            getattr(layer, name).copy_(value.to(device))
    return layer


def pipeline_named_params(layers: Sequence[Any]) -> Dict[str, torch.nn.Parameter]:
    named = {}
    for li, layer in enumerate(layers):
        for pname, p in layer.named_parameters():
            named[f"{li}.{pname}"] = p
    return named


def make_sparse_tensor(impl: SimpleNamespace, inp: Dict[str, Any], device: str,
                       requires_grad: bool = False) -> Tuple[Any, torch.Tensor]:
    feats = inp["features"].to(device).clone()
    if requires_grad:
        feats.requires_grad_(True)
    x = impl.pytorch.SparseConvTensor(feats, inp["indices"].to(device),
                                      list(inp["spatial_shape"]),
                                      int(inp["batch_size"]))
    return x, feats


def _add_input_features(out_indices: torch.Tensor, spatial_shape: Sequence[int],
                        channels: int, dtype: torch.dtype) -> torch.Tensor:
    """Deterministic, order-covariant residual features keyed on coordinates."""
    lin = linearize_indices(out_indices.cpu(), spatial_shape).to(torch.float64)
    base = (lin % 97).to(torch.float64) / 97.0 - 0.5
    chan = torch.arange(channels, dtype=torch.float64) / max(channels, 1)
    feats = base[:, None] * 0.7 + chan[None, :] * 0.3
    return feats.to(dtype)


def run_pipeline_case(impl: SimpleNamespace, case: Dict[str, Any],
                      device: str = DEVICE) -> Dict[str, Any]:
    """Execute a golden pipeline case and return raw results."""
    dtype = getattr(torch, case["dtype"])
    training = bool(case["training"])
    layers = [build_layer(impl, spec, dtype, device) for spec in case["layers"]]
    for layer in layers:
        layer.train(training)

    x, feats_leaf = make_sparse_tensor(impl, case["input"], device,
                                       requires_grad=training)

    if case.get("add_input"):
        assert len(layers) == 1 and not training
        with torch.no_grad():
            pre = layers[0](x)
            add_feats = _add_input_features(pre.indices, pre.spatial_shape,
                                            pre.features.shape[1], dtype)
            add_x = impl.pytorch.SparseConvTensor(
                add_feats.to(device), pre.indices, list(pre.spatial_shape),
                int(case["input"]["batch_size"]))
            x2, _ = make_sparse_tensor(impl, case["input"], device)
            out = layers[0](x2, add_input=add_x)
        return {"out": out, "layers": layers, "feats_leaf": None}

    if not training:
        with torch.no_grad():
            out = x
            for layer in layers:
                out = layer(out)
        return {"out": out, "layers": layers, "feats_leaf": None}

    out = x
    for layer in layers:
        out = layer(out)
    return {"out": out, "layers": layers, "feats_leaf": feats_leaf}


def check_pipeline_case(impl: SimpleNamespace, case: Dict[str, Any],
                        device: str = DEVICE):
    dtype = getattr(torch, case["dtype"])
    res = run_pipeline_case(impl, case, device)
    out, layers = res["out"], res["layers"]
    expect = case["expect"]

    if case.get("returns_dense"):
        actual = out.detach().cpu() if isinstance(out, torch.Tensor) else out
        assert_close(actual, expect["out_features"], expect["atol_out"],
                     f"{case['id']}: dense output")
        order = None
    else:
        order = assert_sparse_output(out, expect, dtype, msg=case["id"])

    if not case["training"]:
        return

    # Backward with a coordinate-aligned upstream gradient.
    grad_canon = expect["grad_out"].to(device)
    if case.get("returns_dense"):
        grad_rows = grad_canon
        out_feats = out
    else:
        inv = inverse_permutation(order.to(device))
        grad_rows = grad_canon[inv]
        out_feats = out.features
    out_feats.backward(grad_rows.to(out_feats.dtype))

    feats_leaf = res["feats_leaf"]
    assert feats_leaf.grad is not None, f"{case['id']}: no input gradient"
    assert_close(feats_leaf.grad, expect["grad_input"],
                 expect["atol_grad_input"], f"{case['id']}: grad_input")

    named = pipeline_named_params(layers)
    assert set(named.keys()) == set(expect["grad_params"].keys()), (
        f"{case['id']}: parameter set mismatch: "
        f"{sorted(named)} != {sorted(expect['grad_params'])}")
    for key, ginfo in expect["grad_params"].items():
        param = named[key]
        assert param.grad is not None, f"{case['id']}: {key} has no grad"
        assert_close(param.grad, ginfo["grad"], ginfo["atol"],
                     f"{case['id']}: {key}.grad")


# ---------------------------------------------------------------------------
# Reference network (family F)
# ---------------------------------------------------------------------------

def build_unet3d(impl: SimpleNamespace, in_channels: int = 6, base: int = 16):
    """Small UNet exercising subm/regular/inverse convs, indice_key sharing,
    SparseSequential with dense modules, JoinTable and the conv1x1 path."""
    sp = impl.pytorch
    nn = torch.nn

    class UNet3d(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = sp.SparseSequential(
                sp.SubMConv3d(in_channels, base, 3, padding=1,
                              indice_key="s0", bias=False),
                nn.BatchNorm1d(base),
                nn.ReLU(),
                sp.SubMConv3d(base, base, 3, padding=1, indice_key="s0",
                              bias=False),
            )
            self.down = sp.SparseConv3d(base, base * 2, 3, stride=2,
                                        padding=1, indice_key="d0",
                                        bias=False)
            self.mid = sp.SparseSequential(
                sp.SubMConv3d(base * 2, base * 2, 3, padding=1,
                              indice_key="s1", bias=False),
                nn.BatchNorm1d(base * 2),
                nn.ReLU(),
            )
            self.up = sp.SparseInverseConv3d(base * 2, base, 3,
                                             indice_key="d0", bias=False)
            self.join = sp.JoinTable()
            self.head = sp.SubMConv3d(base * 2, 8, 1, bias=True)

        def forward(self, x):
            a = self.stem(x)
            b = self.down(a)
            c = self.mid(b)
            d = self.up(c)
            e = self.join([d, a])
            return self.head(e)

    return UNet3d()


# ---------------------------------------------------------------------------
# ops-level canonical forms
# ---------------------------------------------------------------------------

def canon_native_pairs(pair: torch.Tensor, indice_num_per_loc: torch.Tensor
                       ) -> List[torch.Tensor]:
    """Per-offset sorted (in, out) pair lists for a Native [2, kv, N] pair tensor."""
    pair = pair.detach().cpu().long()
    npl = indice_num_per_loc.detach().cpu().long()
    out = []
    for k in range(pair.shape[1]):
        n = int(npl[k])
        pk = pair[:, k, :n].t().contiguous()  # [n, 2] (in, out)
        if n > 0:
            key = pk[:, 0] * (pk.max() + 1) + pk[:, 1]
            pk = pk[torch.argsort(key)]
        out.append(pk)
    return out


def canon_igemm_pairs(out_indices: torch.Tensor, spatial_shape: Sequence[int],
                      pair_fwd: torch.Tensor, pair_bwd: Optional[torch.Tensor]
                      ) -> Dict[str, torch.Tensor]:
    """Canonicalize implicit-gemm pair tables against output-row permutation.

    pair_fwd [kv, M]: values are input rows (fixed order) -> permute columns.
    pair_bwd [kv, N]: values are output rows -> remap values through inverse perm.
    """
    order = canonical_order(out_indices.cpu(), spatial_shape)
    inv = inverse_permutation(order)
    pf = pair_fwd.detach().cpu().long()[:, order]
    res = {"pair_fwd": pf}
    if pair_bwd is not None and pair_bwd.numel() > 0:
        pb = pair_bwd.detach().cpu().long()
        pb = torch.where(pb >= 0, inv[pb.clamp(min=0)], pb)
        res["pair_bwd"] = pb
    return res


def check_mask_properties(pair_fwd_or_bwd: torch.Tensor,
                          pair_mask: torch.Tensor,
                          mask_argsort: torch.Tensor):
    """Property checks for pair_mask / mask_argsort (no golden needed).

    Reference semantics (verified empirically against spconv): the pair-gen
    sorts pair_mask in place, with mask_argsort[i] = original row of the i-th
    sorted entry, so pair_mask[i] == bits(pair[:, mask_argsort[i]]) where
    bit k is set iff pair[k, row] >= 0. (kv <= 32, single split.)
    """
    pair = pair_fwd_or_bwd.detach().cpu().long()
    mask = pair_mask.detach().cpu()
    if mask.dim() == 2 and mask.shape[1] == 1:
        mask = mask[:, 0]
    mask = mask.to(torch.int64) & 0xFFFFFFFF
    args = mask_argsort.detach().cpu().long()
    assert sorted(args.tolist()) == list(range(args.numel())), \
        "mask_argsort is not a permutation"
    assert bool((mask[1:] >= mask[:-1]).all()), \
        "pair_mask is not sorted ascending"
    kv = pair.shape[0]
    if kv <= 32:
        expected = torch.zeros(pair.shape[1], dtype=torch.int64)
        for k in range(kv):
            expected |= (pair[k] >= 0).to(torch.int64) << k
        assert torch.equal(mask, expected[args]), \
            "pair_mask != row bits permuted by mask_argsort"


def canon_voxel_result(indices: torch.Tensor, voxels: torch.Tensor,
                       num_per_voxel: torch.Tensor):
    """Canonical form for Point2Voxel outputs: sort voxels by coordinate and
    sort points inside each voxel lexicographically."""
    idx = indices.detach().cpu().long()
    vox = voxels.detach().cpu().to(torch.float64)
    npv = num_per_voxel.detach().cpu().long()
    if idx.shape[0] == 0:
        return idx, vox, npv
    maxc = int(idx.max()) + 2
    key = torch.zeros(idx.shape[0], dtype=torch.int64)
    for d in range(idx.shape[1]):
        key = key * maxc + idx[:, d]
    order = torch.argsort(key)
    idx, vox, npv = idx[order], vox[order], npv[order]
    vox_sorted = vox.clone()
    for i in range(vox.shape[0]):
        n = int(min(npv[i], vox.shape[1]))
        pts = vox[i, :n]
        pkey = pts[:, 0].clone()
        for d in range(1, pts.shape[1]):
            pkey = pkey * 131071.0 + pts[:, d]
        vox_sorted[i, :n] = pts[torch.argsort(pkey)]
    return idx, vox_sorted, npv
