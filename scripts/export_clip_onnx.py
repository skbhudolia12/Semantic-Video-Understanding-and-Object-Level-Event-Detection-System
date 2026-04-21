"""
ONNX export helper for CLIP ViT-B/32
======================================
Run this script ONCE before using the optimised Tier-2 scanner:

  python scripts/export_clip_onnx.py

What it does
------------
  1. Loads the standard OpenAI CLIP ViT-B/32 model via the ``clip`` package.
  2. Exports only the **image encoder** to ONNX (text encoder stays in PyTorch
     because it is only called a handful of times per frame, not in the
     sliding-window hot path).
  3. Runs a cosine-similarity sanity check: the exported ONNX output must
     match the PyTorch output with cosine similarity > 0.999.
  4. Saves the file to ``models/clip_vit_b32.onnx``.

After exporting, ``OnnxClipEncoder`` (src/models/onnx_clip_encoder.py) will
automatically detect and use the ONNX file on next pipeline run.

Requirements
------------
  pip install clip-by-openai onnx onnxruntime torch torchvision
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Project root → so we can import src.*
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_DEFAULT_OUT = _PROJECT_ROOT / "models" / "clip_vit_b32.onnx"


def export_clip_image_encoder(out_path: Path, opset: int = 14) -> None:
    """Export the CLIP ViT-B/32 image encoder to ONNX."""
    import clip  # openai/CLIP

    logger.info("Loading CLIP ViT-B/32 from OpenAI...")
    device = "cpu"  # export on CPU for maximum portability
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # Dummy input — standard CLIP 224×224 preprocessed image
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    # We only export the visual (image) encoder
    visual = model.visual

    logger.info("Exporting image encoder to ONNX (opset=%d)...", opset)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        visual,
        dummy_input,
        str(out_path),
        opset_version=opset,
        input_names=["pixel_values"],
        output_names=["image_features"],
        dynamic_axes={
            "pixel_values":   {0: "batch_size"},
            "image_features": {0: "batch_size"},
        },
        do_constant_folding=True,
    )
    logger.info("Saved: %s  (%.1f MB)", out_path, out_path.stat().st_size / 1e6)


def verify_onnx(out_path: Path) -> float:
    """
    Compare ONNX output to PyTorch output on a random batch.
    Returns the minimum cosine similarity across the batch.
    """
    import clip
    import onnxruntime as ort

    logger.info("Verifying ONNX export...")
    device = "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()

    dummy = torch.randn(4, 3, 224, 224, device=device)  # batch of 4

    # PyTorch reference
    with torch.no_grad():
        pt_out = model.visual(dummy).numpy().astype(np.float32)

    # ONNX output
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ort_out = session.run(
        ["image_features"], {"pixel_values": dummy.numpy().astype(np.float32)}
    )[0]

    # Cosine similarity row-wise
    pt_norm  = pt_out  / np.linalg.norm(pt_out,  axis=1, keepdims=True)
    ort_norm = ort_out / np.linalg.norm(ort_out, axis=1, keepdims=True)
    sims = (pt_norm * ort_norm).sum(axis=1)
    min_sim = float(sims.min())
    logger.info("  Cosine similarity (min over batch): %.6f", min_sim)
    return min_sim


def main():
    parser = argparse.ArgumentParser(description="Export CLIP image encoder to ONNX")
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Output ONNX file path (default: {_DEFAULT_OUT})",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip cosine-similarity verification after export",
    )
    args = parser.parse_args()

    export_clip_image_encoder(args.out, opset=args.opset)

    if not args.skip_verify:
        min_sim = verify_onnx(args.out)
        if min_sim < 0.999:
            logger.error(
                "Verification FAILED (min cosine similarity %.4f < 0.999). "
                "The ONNX export may be incorrect.",
                min_sim,
            )
            sys.exit(1)
        else:
            logger.info(
                "Verification PASSED ✓  (min cosine similarity=%.6f)", min_sim
            )

    logger.info("Done. OnnxClipEncoder will use this file automatically.")


if __name__ == "__main__":
    main()
