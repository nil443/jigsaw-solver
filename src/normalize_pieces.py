#!/usr/bin/env python3
# normalize_pieces.py
#
# Rotate each piece to the correct multiple of 90°, crop the excess canvas
# and save the normalized pieces. Exports a `normalize()` function so
# it can be called from other modules, while still supporting the
# command-line interface.
#
# CLI Usage:
#   python normalize_pieces.py -i out_pieces -o pieces
#
# The normalize() function returns a dict {png_name: applied_angle}

from __future__ import annotations
import os
import glob
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


# ─────────────────────────────────────────────────
def normalize_image(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Rotate the piece to align it vertically (multiples of 90°) and crop
    the extra transparent area. Returns (normalized_image, applied_angle_in_degrees).
    """
    # — separate alpha channel —
    if img.shape[2] == 4:
        bgr, alpha = img[:, :, :3], img[:, :, 3]
    else:
        bgr = img
        alpha = np.full(bgr.shape[:2], 255, dtype=np.uint8)

    # — binary mask —
    _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img, 0.0

    cnt = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    angle = rect[-1]

    # Adjust so the longer side becomes horizontal
    if rect[1][0] < rect[1][1]:
        angle += 90

    # — prevent cropping during rotation —
    h, w = bgr.shape[:2]
    diag = int(np.sqrt(h**2 + w**2))
    pad_y, pad_x = (diag - h) // 2, (diag - w) // 2
    bgr_pad = cv2.copyMakeBorder(bgr, pad_y, pad_y, pad_x, pad_x,
                                 cv2.BORDER_CONSTANT, value=(255, 255, 255))
    alpha_pad = cv2.copyMakeBorder(alpha, pad_y, pad_y, pad_x, pad_x,
                                   cv2.BORDER_CONSTANT, value=0)

    # — rotate —
    center = (bgr_pad.shape[1] // 2, bgr_pad.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    bgr_rot = cv2.warpAffine(bgr_pad, M, (bgr_pad.shape[1], bgr_pad.shape[0]),
                             flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    alpha_rot = cv2.warpAffine(alpha_pad, M, (alpha_pad.shape[1], alpha_pad.shape[0]),
                               flags=cv2.INTER_NEAREST, borderValue=0)

    # — crop to the piece —
    ys, xs = np.where(alpha_rot > 10)
    if ys.size == 0 or xs.size == 0:
        merged = cv2.merge([bgr_rot, alpha_rot])
    else:
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        merged = cv2.merge([bgr_rot[y0:y1+1, x0:x1+1],
                            alpha_rot[y0:y1+1, x0:x1+1]])

    return merged, angle


# ─────────────────────────────────────────────────
def normalize(in_dir: str = "out_pieces", out_dir: str = "pieces") -> Dict[str, float]:

    in_path = Path(in_dir)
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)

    rotations: Dict[str, float] = {}

    for file in sorted(glob.glob(str(in_path / "*.png"))):
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"⚠️  Cannot read {file}")
            continue

        norm_img, ang = normalize_image(img)
        name = os.path.basename(file)
        destination = out_path / name
        cv2.imwrite(str(destination), norm_img)
        rotations[name] = ang
        print(f"✔︎ {name:20} → rotated {ang:6.2f}°   saved to {destination}")

    # Summary
    print("\nRotation summary:")
    for name, angle in rotations.items():
        print(f"  · {name:<20} {angle:6.2f}°")

    return rotations


# ───────────────────────── CLI ───────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(
        description="Normalize piece orientations to multiples of 90°"
    )
    p.add_argument("-i", "--input",  default="out_pieces",
                   help="Folder with the original pieces")
    p.add_argument("-o", "--output", default="pieces",
                   help="Folder to save the normalized pieces")
    p.add_argument("--save-json", action="store_true",
                   help="Save a rotations.json file with the angles")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    rot = normalize(args.input, args.output)

    if args.save_json:
        json_path = Path(args.output) / "rotations.json"
        with open(json_path, "w") as f:
            json.dump(rot, f, indent=2)
        print(f"\n💾  Rotations saved to {json_path}")
