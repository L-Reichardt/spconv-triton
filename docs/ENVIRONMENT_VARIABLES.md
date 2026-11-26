# Environment variables

spconv-Triton reads four environment variables. Only one of them is useful for most users. The other three exist for compatibility with spconv-cuda, the CUDA build of spconv. We recommend that you keep the defaults.

| Variable | Default | Effect |
|---|---|---|
| `SPCONV_ALLOW_FP16_ACCUM` | `0` | `1` makes fp16 GEMMs accumulate in fp16, as spconv-cuda does. This is faster only on consumer Ampere and Ada GPUs. It also loses some accuracy, so the default is off. |
| `SPCONV_DO_SORT` | `1` | `0` skips the pair-mask sort in implicit-GEMM pair generation. This is a performance option inherited from spconv-cuda. |
| `SPCONV_FX_TRACE_MODE` | `0` | `1` makes `SparseConvTensor` skip shape assertions. This lets you trace a model with `torch.fx`. The option is inherited from spconv-cuda. |
| `SPCONV_SAVED_WEIGHT_LAYOUT` | unset | Weight layout of the checkpoint to load: `KRSC`, `RSKC`, or `RSCK`. This matches spconv-cuda, including its bug. Loading a non-KRSC layout always fails. |

## Removed

These variables are no longer read: `SPCONV_DISABLE_JIT`, `SPCONV_DEBUG_SAVE_PATH`, `SPCONV_BWD_SPLITK`, and `SPCONV_INT8_DEBUG`. They controlled the JIT build, debug dumps, split-K tuning, and int8 paths of spconv-cuda. These paths do not exist in spconv-Triton. The module constants with the same names still exist, so existing imports work.

## Related: constants

`spconv_triton.constants.SPCONV_ALLOW_TF32` is not an environment variable. It is a module constant, as in spconv-cuda. Its default is `False`. Set it to `True` to allow TF32 in the fp32 Triton GEMMs.

Some constants have no effect. See the comments in `spconv_triton/constants.py`.

## Related: Triton cache

Set `TRITON_CACHE_DIR` to a stable path. Then Triton keeps compiled kernels between runs. See the README for details.

## Test-suite variables

These variables affect only the frozen test suite in `tests/`. They do not affect the installed package.

| Variable | Default | Effect |
|---|---|---|
| `SPCONV_TEST_IMPL` | `spconv_triton` | Implementation to test. `spconv` runs the suite against the unmaintained spconv reference. Only the `benchmark` tox environment installs spconv. |
| `SPCONV_TEST_EXPECT_WARM` | unset | `1` enables the warm-cache check in `tests/conftest.py` and `tests/test_zz_warm_cache.py`. The whole run must reuse an existing `TRITON_CACHE_DIR`, and no kernel may compile again. The `warm` tox environment sets this variable. The `cold` environment must fill the shared cache first. Run `tox -m warmcold` to do both. |
