# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.1.0b1]

First public beta.

### 2026-07-08 (Initial Release)

- All spconv 2.3.8 layer types, forward and backward, 1d–3d.
- FP32 (IEEE), TF32, and FP16 precision paths.
- `python 3.10 – 3.14`, `torch 2.4 – 2.12` support (torch brings its matching triton).
- Basic torch.compile support through layerwise eager execution.
- Verified on NVIDIA Ampere (RTX 3060) and AMD (MI300X) hardware.
- Verified through inference (Utonia, Cylinder3D) parity and training (Uni3DETR)

### 2026-07-09 (Testing)

- Updated tox and verified `torch 2.13`
- Verified Nvidia T4, A100, H100, L4, B200

## [0.1.0b2]

### Test suite

- Slimmed the test suite to decisive parts
- Added missing ROCm env var to allow TF32 (which is supported from torch 2.7+ onward)
- Added test suite to git
- Outsourced benchmarking into a dedicated environment for comparisons with other libraries
- Testing distributed training