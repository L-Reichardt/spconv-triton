"""Byte-identity check: fused candidate kernel vs the original torch loop.

Reproduces the exact pre-fusion per-dimension loop as a reference and compares
`valid` (everywhere) and `out_lin` (on valid entries -- the only ones consumed
downstream) plus the resulting unique-count, across subm / regular / transpose,
ndim 1..4, stride variations, and both the int32 and int64 linearization paths.

Also verifies the fused implicit-gemm SCATTER kernel
(`pair_kernels.scatter_igemm_pairs`, Kernel C) bit-exact against the torch
`nonzero` + `index_put_` scatter it replaces, for the same case matrix plus
kv=1 and a num_out_act_bound truncation, and asserts the forward scatter is
collision-free (the injectivity property the direct-store fusion relies on).
"""

import functools
import itertools

import torch

from spconv_triton.pytorch._impl import pair_kernels
from spconv_triton.pytorch._impl import pairs as P

DEV = "cuda"


def lin_in(idx_lin, shape):
    """Batch-major linearization of input coords (mirrors pairs._linearize)."""
    b = idx_lin[:, 0]
    coords = idx_lin[:, 1:]
    lin = b
    for i, s in enumerate(shape):
        lin = lin * int(s) + coords[:, i]
    return lin


def ref_membership(out_lin, sorted_lin, sort_idx, prior_valid, N):
    """The original torch subm membership chain (verbatim semantics)."""
    pos = torch.searchsorted(sorted_lin, out_lin.reshape(-1))
    pos_c = pos.clamp(max=max(N - 1, 0))
    found = (sorted_lin[pos_c] == out_lin.reshape(-1)) & (pos < N)
    out_row = sort_idx[pos_c].reshape(out_lin.shape)
    valid = prior_valid & found.reshape(out_lin.shape)
    return out_row, valid


def check_membership(label, idx_lin, out_lin, k_valid, out_shape):
    """Bit-exact check of Kernel A (subm membership) vs the torch chain."""
    N = idx_lin.shape[0]
    in_lin = lin_in(idx_lin, out_shape)
    sorted_lin, sort_idx = torch.sort(in_lin)
    r_or, r_v = ref_membership(out_lin, sorted_lin, sort_idx, k_valid, N)
    k_or, k_v = pair_kernels.subm_membership(out_lin, sorted_lin, sort_idx, k_valid)
    # valid bit-exact everywhere; out_row only on valid entries (don't-care else).
    ok = bool(torch.equal(r_v, k_v)) and bool(torch.equal(r_or[r_v], k_or[k_v]))
    flag = "OK " if ok else "FAIL"
    print(f"  [{flag}] membership: {label:<25s} (n_found={int(r_v.sum())})")
    return ok


def ref_scatter(valid, out_row, n_out, N, need_bwd):
    """The original torch back-half scatter (verbatim semantics)."""
    device = valid.device
    kv = valid.shape[0]
    pair_fwd = torch.full((kv, n_out), -1, dtype=torch.int32, device=device)
    pair_bwd = (
        torch.full((kv, N), -1, dtype=torch.int32, device=device)
        if need_bwd
        else torch.Tensor()
    )
    if N > 0:
        vflat = valid.reshape(-1).nonzero(as_tuple=True)[0]
        kk = vflat.div(N, rounding_mode="floor")
        jj = vflat - kk * N
        oro = out_row.reshape(-1)[vflat]
        pair_fwd.index_put_((kk, oro), jj.to(torch.int32))
        if need_bwd:
            pair_bwd.index_put_((kk, jj), oro.to(torch.int32))
    return pair_fwd, pair_bwd


def kern_scatter(valid, out_row, n_out, N, need_bwd):
    """The fused Triton scatter (Kernel C)."""
    device = valid.device
    kv = valid.shape[0]
    pair_fwd = torch.full((kv, n_out), -1, dtype=torch.int32, device=device)
    pair_bwd = (
        torch.full((kv, N), -1, dtype=torch.int32, device=device)
        if need_bwd
        else torch.Tensor()
    )
    pair_kernels.scatter_igemm_pairs(valid, out_row, pair_fwd, pair_bwd, need_bwd)
    return pair_fwd, pair_bwd


def no_collision(valid, out_row, n_out):
    """Each valid (k, out_row) cell is written at most once (injectivity)."""
    kv, N = valid.shape
    k_idx = torch.arange(kv, device=valid.device).unsqueeze(1).expand(kv, N)
    cells = (k_idx.to(torch.int64) * n_out + out_row.to(torch.int64))[valid]
    return int(cells.numel()) == int(torch.unique(cells).numel())


def check_scatter(label, cand, N, n_out=None, valid=None):
    """Bit-exact + collision-free check of Kernel C against the torch scatter."""
    valid = cand.valid if valid is None else valid
    out_row = cand.out_row
    n_out = cand.n_out if n_out is None else n_out
    ok = no_collision(valid, out_row, n_out)
    for need_bwd in (True, False):
        rf, rb = ref_scatter(valid, out_row, n_out, N, need_bwd)
        kf, kb = kern_scatter(valid, out_row, n_out, N, need_bwd)
        ok &= bool(torch.equal(rf, kf))
        if need_bwd:
            ok &= bool(torch.equal(rb, kb))
    flag = "OK " if ok else "FAIL"
    print(f"  [{flag}] scatter: {label:<28s} (kv={cand.kv}, n_out={n_out})")
    return ok


def _prod(v):
    return functools.reduce(lambda a, b: a * b, v, 1)


def kernel_offsets(ksize, device, dtype):
    grids = torch.meshgrid(
        *[torch.arange(k, device=device, dtype=torch.int64) for k in ksize],
        indexing="ij",
    )
    return torch.stack([g.reshape(-1) for g in grids], 1).to(dtype)


def ref_loop(idx_lin, offs, pad, st, dilation, out_shape, kv, transpose):
    """The original per-dimension torch loop (verbatim semantics)."""
    device = idx_lin.device
    N = idx_lin.shape[0]
    ndim = len(out_shape)
    coords = idx_lin[:, 1:]
    b = idx_lin[:, 0]
    valid = torch.ones((kv, N), dtype=torch.bool, device=device)
    out_lin = b.unsqueeze(0).expand(kv, N).contiguous()
    for d in range(ndim):
        off_d = (offs[:, d] * dilation[d]).unsqueeze(1)
        if transpose:
            o_d = coords[:, d].unsqueeze(0) * st[d] + off_d - pad[d]
            valid &= (o_d >= 0) & (o_d < out_shape[d])
        else:
            o_d = coords[:, d].unsqueeze(0) + pad[d] - off_d
            if st[d] == 1:
                valid &= (o_d >= 0) & (o_d < out_shape[d])
            else:
                q = torch.div(o_d, st[d], rounding_mode="floor")
                valid &= (o_d >= 0) & (o_d - q * st[d] == 0) & (q < out_shape[d])
                o_d = q
        out_lin = out_lin * out_shape[d] + o_d
    return out_lin, valid


def make_indices(N, ss, bs, seed):
    # Unique voxel coords per batch -- the production SparseConvTensor invariant
    # (matches gen_input / make_pointcloud). The scatter's collision-free property
    # holds exactly on unique input coords; duplicates would alias output rows.
    g = torch.Generator().manual_seed(seed)
    rows = []
    for b in range(bs):
        c = torch.stack([torch.randint(0, s, (N,), generator=g) for s in ss], 1)
        c = torch.unique(c, dim=0)
        rows.append(torch.cat([torch.full((c.shape[0], 1), b), c], 1))
    return torch.cat(rows).int().to(DEV)


def out_shape_conv(ss, ksize, stride, pad, dil):
    return [
        (ss[i] + 2 * pad[i] - dil[i] * (ksize[i] - 1) - 1) // stride[i] + 1
        for i in range(len(ss))
    ]


def out_shape_deconv(ss, ksize, stride, pad, dil, opad):
    return [
        (ss[i] - 1) * stride[i] - 2 * pad[i] + ksize[i] + opad[i]
        for i in range(len(ss))
    ]


def run_case(label, ss, ksize, stride, pad, dil, subm, transpose, N=4000, bs=2, seed=7):
    ndim = len(ss)
    idx = make_indices(N, ss, bs, seed)
    max_lin = bs * _prod(
        out_shape_deconv(ss, ksize, stride, pad, dil, [0] * ndim)
        if transpose
        else (ss if subm else out_shape_conv(ss, ksize, stride, pad, dil))
    )
    if subm:
        max_lin = max(max_lin, bs * _prod(ss))
    lin_dtype = torch.int32 if max_lin < 2**31 else torch.int64

    if subm:
        eff_pad = [(ksize[i] // 2) * dil[i] for i in range(ndim)]
        eff_st = [1] * ndim
        out_shape = list(ss)
    elif transpose:
        eff_pad, eff_st = list(pad), list(stride)
        out_shape = out_shape_deconv(ss, ksize, stride, pad, dil, [0] * ndim)
    else:
        eff_pad, eff_st = list(pad), list(stride)
        out_shape = out_shape_conv(ss, ksize, stride, pad, dil)

    kv = _prod(ksize)
    idx_lin = idx.to(lin_dtype).contiguous()
    offs = kernel_offsets(ksize, idx.device, lin_dtype)

    ref_lin, ref_valid = ref_loop(
        idx_lin, offs, eff_pad, eff_st, dil, out_shape, kv, transpose
    )
    k_lin, k_valid = pair_kernels.candidate_out_lin_valid(
        idx_lin, offs, eff_pad, eff_st, list(dil), out_shape, kv, transpose
    )

    valid_eq = torch.equal(ref_valid, k_valid)
    lin_eq = torch.equal(ref_lin[ref_valid], k_lin[k_valid])
    # downstream unique-count must match
    ru = torch.unique(ref_lin[ref_valid])
    ku = torch.unique(k_lin[k_valid])
    nout_eq = ru.numel() == ku.numel() and torch.equal(ru, ku)

    ok = valid_eq and lin_eq and nout_eq
    flag = "OK " if ok else "FAIL"
    print(
        f"  [{flag}] {label:<34s} dtype={lin_dtype!s:<12s} "
        f"valid={valid_eq} lin={lin_eq} nout={nout_eq} "
        f"(n_valid={int(ref_valid.sum())}, n_out={ru.numel()})"
    )

    # Kernel C: scatter the implicit-gemm tables through the production
    # candidate path, bit-exact vs the torch scatter + collision-free check.
    cand = P.compute_candidates(
        idx,
        bs,
        list(ss),
        list(ksize),
        list(stride),
        list(pad),
        list(dil),
        [0] * ndim,
        subm,
        transpose,
    )
    ok &= check_scatter(label, cand, idx.shape[0])
    # Kernel A: subm membership bit-exact vs the torch searchsorted chain.
    if subm:
        ok &= check_membership(label, idx_lin, k_lin, k_valid, out_shape)
    # num_out_act_bound truncation (regular only): drop the upper output rows
    # and re-mask validity, exactly as igemm_pairs does.
    if not subm and cand.n_out > 4:
        n_b = cand.n_out // 2
        valid_b = cand.valid & (cand.out_row < n_b)
        ok &= check_scatter(
            f"{label} bound", cand, idx.shape[0], n_out=n_b, valid=valid_b
        )
    return ok


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    all_ok = True
    # ndim 1..4, subm (odd k), stride 1
    for ndim in (1, 2, 3, 4):
        ss = tuple([12, 16, 20, 8][:ndim])
        k = tuple([3] * ndim)
        all_ok &= run_case(
            f"subm {ndim}d k3", ss, k, [1] * ndim, [1] * ndim, [1] * ndim, True, False
        )

    # regular conv: stride 1 & 2, padding, dilation, ndim 2/3
    for ndim, stride in itertools.product((2, 3), (1, 2)):
        ss = tuple([24, 28, 20][:ndim])
        k = tuple([3] * ndim)
        all_ok &= run_case(
            f"conv {ndim}d k3 s{stride}",
            ss,
            k,
            [stride] * ndim,
            [1] * ndim,
            [1] * ndim,
            False,
            False,
        )
    # mixed stride, even kernel, dilation
    all_ok &= run_case(
        "conv 3d k2 s[2,1,2]",
        (24, 28, 20),
        (2, 2, 2),
        [2, 1, 2],
        [0, 0, 0],
        [1, 1, 1],
        False,
        False,
    )
    all_ok &= run_case(
        "conv 3d k3 dil2 pad2",
        (32, 32, 32),
        (3, 3, 3),
        [1, 1, 1],
        [2, 2, 2],
        [2, 2, 2],
        False,
        False,
    )
    all_ok &= run_case(
        "conv 2d k5 s3 p1", (40, 40), (5, 5), [3, 3], [1, 1], [1, 1], False, False
    )

    # transpose (deconv), ndim 2/3, stride 1 & 2
    for ndim, stride in itertools.product((2, 3), (1, 2)):
        ss = tuple([16, 20, 12][:ndim])
        k = tuple([3] * ndim)
        all_ok &= run_case(
            f"deconv {ndim}d k3 s{stride}",
            ss,
            k,
            [stride] * ndim,
            [1] * ndim,
            [1] * ndim,
            False,
            True,
        )

    # int64 path: huge spatial volume forces int64 linearization
    all_ok &= run_case(
        "subm 3d k3 BIG(int64)",
        (1024, 1024, 1024),
        (3, 3, 3),
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        True,
        False,
        N=2000,
    )
    all_ok &= run_case(
        "conv 3d k3 s2 BIG(int64)",
        (1100, 1100, 1100),
        (3, 3, 3),
        [2, 2, 2],
        [1, 1, 1],
        [1, 1, 1],
        False,
        False,
        N=2000,
    )

    # larger N (representative)
    all_ok &= run_case(
        "subm 3d k3 N~90k",
        (64, 64, 64),
        (3, 3, 3),
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        True,
        False,
        N=45_000,
    )
    all_ok &= run_case(
        "conv 3d k3 s2 N~90k",
        (64, 64, 64),
        (3, 3, 3),
        [2, 2, 2],
        [1, 1, 1],
        [1, 1, 1],
        False,
        False,
        N=45_000,
    )

    # kv=1 (its own autotune/codegen specialization; reaches pair gen via the
    # strided-conv1x1 / get_indice_pairs path).
    all_ok &= run_case(
        "conv 3d k1 s2 (kv=1)",
        (32, 32, 32),
        (1, 1, 1),
        [2, 2, 2],
        [0, 0, 0],
        [1, 1, 1],
        False,
        False,
    )

    print("\n" + ("ALL BYTE-IDENTICAL ✅" if all_ok else "MISMATCH DETECTED ❌"))
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
