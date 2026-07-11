"""Golden data generation for the WIDE-CHANNEL suite, from UNCHANGED spconv.

Covers C=128/256 conv widths - the production-backbone region untested by the
original frozen suites (which topped out at C=64 fp32 / C=32 fp16). A GEMM
block/tail bug above C=64 would pass every other test; these cases pin it.

Run once, with the reference implementation, on a GPU:

    SPCONV_TEST_IMPL=spconv uv run python tests_wide/gen_golden_wide.py

Reuses the frozen generator machinery:
  * fp32 cases -> tests/gen_golden.py (noise-calibrated atol, the fp32 contract)
  * fp16 cases -> tests_fp16/gen_golden_fp16.py make_case (fp32-reference
    bounding: atol = 3 x reference's own fp16-vs-fp32 deviation + 1e-3)
Both generators assert the reference implementation at import time.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
sys.path.insert(0, str(Path(__file__).parent.parent / "tests_fp16"))
sys.path.insert(0, str(Path(__file__).parent))

import gen_golden as G
import gen_golden_fp16 as F

WIDE_DATA_DIR = Path(__file__).parent / "data"
WIDE_DATA_DIR.mkdir(parents=True, exist_ok=True)


def L(cls, **ctor):
    return {"cls": cls, "ctor": ctor}


def pipeline_case(cid, layers, inp, alt="Native", repeats=1):
    """fp32 pipeline case via the frozen finalize_case (noise-calibrated)."""
    case = {
        "id": cid,
        "kind": "pipeline",
        "dtype": "float32",
        "training": True,
        "layers": layers,
        "input": inp,
        "add_input": False,
    }
    G.gen_params(cid, layers, "float32")
    algos = (alt,) if alt else ()
    G.finalize_case(case, alt_algos=algos, repeats=repeats)
    refmax = float(case["expect"]["out_features"].abs().max())
    sens = case["expect"]["atol_out"] / max(refmax, 1e-9)
    print(
        f"  [wide] {cid}: out={tuple(case['expect']['out_features'].shape)}"
        f" atol_out={case['expect']['atol_out']:.2e} ({sens * 100:.3f}% of max|ref|)"
    )
    return case


def fp32_cases():
    cases = []
    # SubM at C=128 / C=256: multi-tile GEMM in both the channel-reduction
    # (BLOCK_R) and output-channel (BLOCK_N) dims, fwd + dW + dIn + db.
    cases.append(
        pipeline_case(
            "wide_subm3d_c128",
            [
                L(
                    "SubMConv3d",
                    in_channels=128,
                    out_channels=128,
                    kernel_size=3,
                    padding=1,
                )
            ],
            G.gen_input(
                "wide_subm3d_c128", [32, 32, 32], 2, 128, 1200, mode="clustered"
            ),
        )
    )
    cases.append(
        pipeline_case(
            "wide_subm3d_c256",
            [
                L(
                    "SubMConv3d",
                    in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    padding=1,
                )
            ],
            G.gen_input(
                "wide_subm3d_c256", [32, 32, 32], 2, 256, 1000, mode="clustered"
            ),
        )
    )
    # Regular strided conv with unequal in/out widths at scale.
    cases.append(
        pipeline_case(
            "wide_conv3d_64_128",
            [
                L(
                    "SparseConv3d",
                    in_channels=64,
                    out_channels=128,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            ],
            G.gen_input(
                "wide_conv3d_64_128", [32, 32, 32], 2, 64, 1500, mode="clustered"
            ),
        )
    )
    cases.append(
        pipeline_case(
            "wide_conv3d_128_256",
            [
                L(
                    "SparseConv3d",
                    in_channels=128,
                    out_channels=256,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            ],
            G.gen_input(
                "wide_conv3d_128_256", [32, 32, 32], 2, 128, 1500, mode="clustered"
            ),
        )
    )
    # NON-block-aligned widths above 64: in=136 (=2*64+8) and out=130 (=2*64+2)
    # both leave a tail tile, exercising the GEMM boundary masking that the
    # power-of-two cases above never hit.
    cases.append(
        pipeline_case(
            "wide_subm3d_c136_130_tail",
            [
                L(
                    "SubMConv3d",
                    in_channels=136,
                    out_channels=130,
                    kernel_size=3,
                    padding=1,
                )
            ],
            G.gen_input(
                "wide_subm3d_c136_130_tail", [32, 32, 32], 2, 136, 900, mode="clustered"
            ),
        )
    )
    return cases


def fp16_cases():
    # Tight fp16 bounding (channels must be multiples of 16 for spconv's igemm).
    return [
        F.make_case(
            "wide_fp16_subm3d_c128",
            [F.L("SubMConv3d", in_channels=128, out_channels=128, kernel_size=3)],
            [32, 32, 32],
            2,
            128,
            1200,
        ),
        F.make_case(
            "wide_fp16_conv3d_128_256",
            [
                F.L(
                    "SparseConv3d",
                    in_channels=128,
                    out_channels=256,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            ],
            [32, 32, 32],
            2,
            128,
            1500,
        ),
    ]


def main():
    torch.backends.cudnn.deterministic = True
    print("generating golden_wide_conv.pt ...")
    torch.save({"cases": fp32_cases()}, WIDE_DATA_DIR / "golden_wide_conv.pt")
    print("generating golden_wide_fp16.pt ...")
    torch.save({"cases": fp16_cases()}, WIDE_DATA_DIR / "golden_wide_fp16.pt")
    print("ALL WIDE GOLDEN DATA GENERATED")
    for f in sorted(WIDE_DATA_DIR.iterdir()):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
