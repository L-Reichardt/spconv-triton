"""Subprocess runner for SPCONV_SAVED_WEIGHT_LAYOUT=RSKC behavior parity.

Upstream spconv 2.3.8's load hook double-permutes the weight (once in the
layout block, again in the ALL_WEIGHT_IS_KRSC block), so loading an
RSKC-layout checkpoint ALWAYS fails with a size mismatch. The port
replicates the hook verbatim and must fail identically.

Exits 0 and prints LOAD_RAISED if load_state_dict raises RuntimeError;
exits 1 if the load unexpectedly succeeds.

Reads env: SPCONV_TEST_IMPL (and SPCONV_SAVED_WEIGHT_LAYOUT=RSKC).
"""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
import helpers

assert os.environ.get("SPCONV_SAVED_WEIGHT_LAYOUT") == "RSKC"

impl = helpers.load_impl()

torch.manual_seed(99)
layer = impl.pytorch.SubMConv3d(4, 8, 3, bias=False)

g = torch.Generator().manual_seed(123)
w_krsc = torch.randn(8, 3, 3, 3, 4, generator=g)
# RSKC layout checkpoint: [*ksize, K, C]
w_rskc = w_krsc.permute(1, 2, 3, 0, 4).contiguous()
try:
    layer.load_state_dict({"weight": w_rskc})
except RuntimeError:
    print("LOAD_RAISED")
    sys.exit(0)
print("LOAD_SUCCEEDED")
sys.exit(1)
