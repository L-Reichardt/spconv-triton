#!/usr/bin/env bash
# Per-layer spconv vs spconv_triton benchmark -> docs/<name>.md (fp32/TF32/fp16 ratios).
# Requires a GPU and reference spconv-cu126, installed on demand via `uv run --with`
# (the default env is spconv-free; the install raises if it fails). Usage: ./bench.sh <name> [title]
# <name> = output stem docs/<name>.md; [title] = table heading (default: detected GPU).
set -euo pipefail

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  echo "usage: ./bench.sh <name> [title]" >&2
  echo "  <name>   output file stem -> docs/<name>.md (e.g. a100)" >&2
  echo "  [title]  table heading (default: detected GPU name)" >&2
  exit 1
fi

# cd to project root so uv picks up this project's environment.
cd "$(dirname "$0")"

NAME="$1"
shift
ARGS=(--name "$NAME")
if [[ $# -ge 1 ]]; then
  ARGS+=(--title "$1")
fi

# Persist Triton's compiled kernels across processes; honour caller's if set.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
# A stray system (ROS) pytest plugin breaks import; disable plugin autoload.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

# Warn (don't block) on GPU contention — it skews timing ratios; run idle for clean numbers.
if command -v nvidia-smi >/dev/null 2>&1; then
  busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .) || busy=0
  if [[ "${busy:-0}" -gt 0 ]]; then
    echo "WARNING: ${busy} process(es) already using the GPU; timing ratios may be skewed." >&2
    echo "         Free the GPU for clean numbers, or press Ctrl-C now." >&2
  fi
fi

# Pull the unmaintained reference spconv-cu126 into the ephemeral run env VIA uv
# (`--with` layers it on top of the locked torch without a lock rewrite). If uv
# can't resolve/install it, uv exits non-zero and `set -euo pipefail` aborts here.
# NOTE: this keeps the project's floated torch. For exact golden-provenance parity
# (pinned torch 2.12.0+cu126) use `uvx --with tox-uv tox -e benchmark -- python scripts/bench_layers.py ...`.
exec uv run --with "spconv-cu126>=2.3.8" python scripts/bench_layers.py "${ARGS[@]}"
