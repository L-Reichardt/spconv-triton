# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.1.0b1] - 2026-07-08

First public beta.

### Added

- All spconv 2.3.8 layer types, forward and backward, 1d–4d.
- FP32 (IEEE), TF32, and FP16 precision paths.
- `python 3.10 – 3.14`, `torch 2.4 – 2.12` support (torch brings its matching triton).
- Basic torch.compile support through layerwise eager execution.
- Verified on NVIDIA Ampere (RTX 3060) and AMD (MI300X) hardware.
- Verified through inference (Utonia, Cylinder3D) parity and training (Uni3DETR)
