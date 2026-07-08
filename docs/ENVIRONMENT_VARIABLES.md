# Environment variables

spconv-Triton reads four environment variables.

One might be useful, the other three are kept for maximum compatibility.
Suggestion: leave defaults.

| Variable | Default | Effect |
|---|---|---|
| `SPCONV_ALLOW_FP16_ACCUM` | `0` | `1` = fp16 GEMMs accumulate in fp16 (matches spconv-cuda). Faster only on consumer Ampere/Ada, but with some accuracy loss, hence defaults to off. |
| `SPCONV_DO_SORT` | `1` | `0` = skip the pair-mask sort in implicit-GEMM pair generation (inherited performance knob from spconv-cuda). |
| `SPCONV_FX_TRACE_MODE` | `0` | `1` = `SparseConvTensor` skips shape asserts so models can be traced with `torch.fx` (inherited from spconv-cuda). |
| `SPCONV_SAVED_WEIGHT_LAYOUT` | unset | Layout (`KRSC`/`RSKC`/`RSCK`) of a checkpoint being loaded (same as spconv-cuda, including its bug. Loading non-KRSC layouts always fails). |

## Removed

Removed: `SPCONV_DISABLE_JIT`, `SPCONV_DEBUG_SAVE_PATH`, `SPCONV_BWD_SPLITK`, and
`SPCONV_INT8_DEBUG` are no longer read — they only configured spconv-cuda's JIT
build, debug dumps, split-K sweep, and int8 paths, none of which exist here (the
same-named module constants remain importable for drop-in compatibility).

## Related: Constants

`spconv_triton.constants.SPCONV_ALLOW_TF32` (default `False`) is a module
constant, not an env var (same as spconv-cuda).
Set it to `True` to allow TF32 in the fp32 Triton GEMMs.

Also, some constants are dead. Refer to notes in `spconv_triton/constants.py`.

## Related: Triton Cache

Keep Triton's own `TRITON_CACHE_DIR` stable to persist compiled kernels (see README).
