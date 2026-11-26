# Changelog

This file lists the notable changes to this project.

## [1.0.1]

- Adds support for Python 3.15 and torch 2.15.
- Fixes deprecation warnings.

## [1.0.0]

- First stable release.

## [0.1.0b1]

First public beta.

### 2026-07-08 (Initial Release)

- Adds all spconv 2.3.8 layer types, for forward and backward passes, in 1D to 3D.
- Adds FP32 (IEEE), TF32, and FP16 precision paths.
- Adds support for Python 3.10 to 3.14 and torch 2.4 to 2.12. torch installs its matching Triton.
- Adds basic torch.compile support through layer-by-layer eager execution.
- Verified on NVIDIA Ampere (RTX 3060) and AMD (MI300X) hardware.
- Verified inference parity (Utonia, Cylinder3D) and training (Uni3DETR).

### 2026-07-09 (Testing)

- Updates tox and verifies torch 2.13.
- Verifies NVIDIA T4, A100, H100, L4, and B200.

## [0.1.0b2]

### Test suite

- Reduces the test suite to its essential parts.
- Adds the missing ROCm environment variable that allows TF32. TF32 is supported from torch 2.7 onward.
- Adds the test suite to git.
- Moves benchmarking to a separate environment, for comparisons with other libraries.
- Tests distributed training.
