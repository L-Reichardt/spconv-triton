"""torch.compile interop (PORT-ONLY, skipped for the reference).

Validates spconv_triton's ``@torch.compiler.disable`` on the sparse conv/pool
forwards: a model mixing spconv_triton layers with ordinary dense layers
(BatchNorm / LayerNorm / ReLU / residual add) can be wrapped in
``torch.compile`` and

  1. run without raising (the sparse layers' data-dependent pair generation +
     Triton kernels are opaque to inductor; the decorator makes each sparse
     layer a graph-break boundary instead of letting inductor fail to codegen
     the data-dependent indexing), and
  2. still let inductor compile AND FUSE the surrounding non-spconv layers.

Two dummy residual-bottleneck U-Nets are exercised:
  - variant "bn": BatchNorm1d + conv bias=False
  - variant "ln": LayerNorm   + conv bias=True

Skipped for ``SPCONV_TEST_IMPL=spconv``: the reference has its own CUDA ops and
none of the port-specific behavior under test here (cf. the port-only kernel /
pair-gen determinism tests).
"""

import re

import pytest
import torch
from torch import nn


# --------------------------------------------------------------------------- #
# Dummy residual-bottleneck U-Net (self-contained, parametrized by the         #
# implementation's pytorch module `spc`). norm / relu / residual-add act on    #
# `.features` (dense [N, C]) so they live OUTSIDE the disabled sparse layers    #
# and are visible to inductor.                                                  #
# --------------------------------------------------------------------------- #
def _norm(kind: str, C: int) -> nn.Module:
    if kind == "bn":
        return nn.BatchNorm1d(C)
    if kind == "ln":
        return nn.LayerNorm(C)
    raise ValueError(kind)


class Bottleneck(nn.Module):
    """Two subm convs with a residual connection and ReLUs."""

    def __init__(self, spc, C: int, kind: str, bias: bool, key: str):
        super().__init__()
        self.conv1 = spc.SubMConv3d(C, C, 3, bias=bias, indice_key=key)
        self.norm1 = _norm(kind, C)
        self.conv2 = spc.SubMConv3d(C, C, 3, bias=bias, indice_key=key)
        self.norm2 = _norm(kind, C)

    def forward(self, x):
        identity = x.features
        out = self.conv1(x)
        out = out.replace_feature(torch.relu(self.norm1(out.features)))
        out = self.conv2(out)
        out = out.replace_feature(self.norm2(out.features))
        out = out.replace_feature(torch.relu(out.features + identity))
        return out


class TinyUNet(nn.Module):
    def __init__(self, spc, in_ch=4, num_classes=5, kind="bn", bias=False):
        super().__init__()
        self.stem = spc.SubMConv3d(in_ch, 16, 3, bias=bias, indice_key="stem")
        self.stem_n = _norm(kind, 16)
        self.enc1 = Bottleneck(spc, 16, kind, bias, "b16")
        self.down1 = spc.SparseConv3d(
            16, 32, 3, stride=2, padding=1, bias=bias, indice_key="d1"
        )
        self.enc2 = Bottleneck(spc, 32, kind, bias, "b32")
        self.down2 = spc.SparseConv3d(
            32, 64, 3, stride=2, padding=1, bias=bias, indice_key="d2"
        )
        self.bott = Bottleneck(spc, 64, kind, bias, "b64")
        self.up2 = spc.SparseInverseConv3d(64, 32, 3, bias=bias, indice_key="d2")
        self.dec2 = Bottleneck(spc, 32, kind, bias, "b32d")
        self.up1 = spc.SparseInverseConv3d(32, 16, 3, bias=bias, indice_key="d1")
        self.dec1 = Bottleneck(spc, 16, kind, bias, "b16d")
        self.head = spc.SubMConv3d(16, num_classes, 1, bias=True, indice_key="stem")

    def forward(self, x):
        x = self.stem(x)
        x = x.replace_feature(torch.relu(self.stem_n(x.features)))
        s1 = self.enc1(x)
        s2 = self.enc2(self.down1(s1))
        x = self.bott(self.down2(s2))
        x = self.up2(x)
        x = x.replace_feature(x.features + s2.features)  # skip add
        x = self.dec2(x)
        x = self.up1(x)
        x = x.replace_feature(x.features + s1.features)  # skip add
        x = self.dec1(x)
        x = self.head(x)
        return x


def _make_input(spc, N=2500, spatial=(40, 40, 40), in_ch=4, batches=2, seed=0):
    """Unique coordinates per batch -> deterministic pair generation, so the
    compiled vs eager output comparison is exact up to inductor fp-reassociation
    of the dense ops (no hash-order / duplicate-coordinate nondeterminism)."""
    g = torch.Generator().manual_seed(seed)
    vol = spatial[0] * spatial[1] * spatial[2]
    rows = []
    for b in range(batches):
        lin = torch.randperm(vol, generator=g)[:N]
        z = lin // (spatial[1] * spatial[2])
        y = (lin // spatial[2]) % spatial[1]
        xx = lin % spatial[2]
        rows.append(torch.stack([torch.full((N,), b), z, y, xx], 1))
    coords = torch.cat(rows, 0).to(torch.int32).cuda()
    feats = torch.randn(coords.shape[0], in_ch, generator=g).cuda()
    return spc.SparseConvTensor(feats, coords, list(spatial), batches)


def _canon(sct) -> torch.Tensor:
    """Rows sorted by linearized (batch-major) output coordinate -> order-
    invariant view for comparing two runs."""
    idx = sct.indices.long()
    lin = idx[:, 0].clone()
    for d, s in enumerate(sct.spatial_shape):
        lin = lin * int(s) + idx[:, d + 1]
    return sct.features[torch.argsort(lin)]


_VARIANTS = {
    "bn-nobias": dict(kind="bn", bias=False),
    "ln-bias": dict(kind="ln", bias=True),
}

_SKIP_REF = "port-only: validates spconv_triton's @torch.compiler.disable"


@pytest.mark.parametrize("variant", list(_VARIANTS))
def test_compile_runs_and_fuses(impl, variant):
    """Compiled eval forward: runs, matches eager, and fuses the dense layers."""
    if impl.name == "spconv":
        pytest.skip(_SKIP_REF)
    spc = impl.pytorch
    import torch._dynamo as dynamo
    import torch._inductor.config as ind_cfg
    import torch._inductor.metrics as metrics
    from torch._inductor.utils import run_and_get_code

    cfg = _VARIANTS[variant]
    torch.manual_seed(0)
    model = TinyUNet(spc, kind=cfg["kind"], bias=cfg["bias"]).cuda().eval()

    dynamo.reset()
    metrics.reset()

    # force_disable_caches: make inductor re-codegen every run so the metrics /
    # generated code below reflect THIS compile (a disk-cache hit would skip
    # codegen and silently leave the counters at zero). Scoped via config.patch
    # so the rest of the suite is unaffected.
    with torch.no_grad(), ind_cfg.patch(force_disable_caches=True):
        eager_out = model(_make_input(spc))
        compiled = torch.compile(model)
        # run_and_get_code triggers the compile and returns the generated code.
        compiled_out, codes = run_and_get_code(compiled, _make_input(spc))

    # (1) ran without raising + produced a sane output
    assert isinstance(compiled_out, spc.SparseConvTensor)
    assert compiled_out.features.shape == eager_out.features.shape

    # (2) numerically equivalent to eager (sparse kernels run eagerly in both;
    # only the dense ops are fused, so differences are fp-reassociation only)
    torch.testing.assert_close(
        _canon(compiled_out), _canon(eager_out), atol=1e-3, rtol=1e-3
    )

    # (3) inductor actually compiled the dense layers into Triton kernels (they
    # did not all fall back to eager)
    assert metrics.generated_kernel_count > 0, "no inductor kernels generated"

    # (4) the non-spconv layers were FUSED: a single generated kernel that names
    # ReLU together with its neighbour (the norm, or the residual add) is direct
    # proof that multiple dense ops share one kernel. (The scheduler-level
    # ir_nodes_pre_fusion ratio is not used here: for a forward-only graph the
    # elementwise chains are already fused into one IR node at lowering, so that
    # ratio stays 1:1 even though fusion happened -- the kernel name is the
    # authoritative signal.)
    code = "\n".join(codes)
    norm_tok = "batch_norm" if cfg["kind"] == "bn" else "layer_norm"
    fused_names = re.findall(r"triton_[a-z]+_fused_[a-z0-9_]+?_\d+", code)
    fused_dense = [
        n for n in fused_names if "relu" in n and (norm_tok in n or "add" in n)
    ]
    assert fused_dense, (
        f"expected a fused relu+({norm_tok}|add) kernel; got {sorted(set(fused_names))}"
    )


@pytest.mark.parametrize("variant", list(_VARIANTS))
def test_compile_trains(impl, variant):
    """Compiled training forward+backward runs and yields finite gradients."""
    if impl.name == "spconv":
        pytest.skip(_SKIP_REF)
    spc = impl.pytorch
    import torch._dynamo as dynamo

    cfg = _VARIANTS[variant]
    torch.manual_seed(0)
    model = TinyUNet(spc, kind=cfg["kind"], bias=cfg["bias"]).cuda().train()

    dynamo.reset()
    compiled = torch.compile(model)
    out = compiled(_make_input(spc))
    loss = out.features.float().pow(2).mean()
    loss.backward()

    assert torch.isfinite(loss).item()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all().item() for g in grads), "non-finite gradient"
