#!/usr/bin/env python3
# segment_pieces.py
#
# Extract each puzzle piece from a black background and save them as RGBA PNGs.
# Designed to be used both as an importable module (function `segment`)
# and from the command line.
#
# CLI Usage:
#   python segment_pieces.py -i in/puzzle_with_pieces.png -o out_pieces --save-debug
#
# The `segment()` function returns a list of generated PNG paths and their positions.

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
import argparse


# ──────────────────────────────────────────────────────────────
def segment(img_path: str, out_dir: str = "out_pieces", save_debug: bool = False) -> Tuple[List[str], Dict[int, Tuple[int, int]]]:

    # ─── 1. Load image ─────────────────────────────────────
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    # ─── 2. Create binary mask ─────────────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # ─── 3. Find contours ───────────────────────────────
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ─── 4. Save debug image with contours (optional) ────
    if save_debug:
        debug = img.copy()
        cv2.drawContours(debug, contours, -1, (0, 255, 0), 2)
        debug_path = Path(out_dir).with_name("detected_contours.png")
        debug_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(debug_path), debug)

    # ─── 5. Extract pieces + record positions ────────────
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []
    positions: Dict[int, Tuple[int, int]] = {}

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        piece = img[y : y + h, x : x + w]
        piece_mask = mask[y : y + h, x : x + w]

        piece_rgba = cv2.cvtColor(piece, cv2.COLOR_BGR2BGRA)
        piece_rgba[:, :, 3] = piece_mask

        fname = Path(out_dir) / f"piece_{i}.png"
        cv2.imwrite(str(fname), piece_rgba)
        paths.append(str(fname))

        # Center of the piece (coordinates in the original image)
        xc = int(x + w / 2)
        yc = int(y + h / 2)
        positions[i] = (xc, yc)

    return paths, positions

# ───────────────────────── CLI ───────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segment puzzle pieces from a black background."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to in/puzzle_with_pieces.png"
    )
    parser.add_argument(
        "-o", "--output", default="out_pieces", help="Folder to save the pieces"
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save image with detected contours",
    )

    args = parser.parse_args()
    segment(args.input, args.output, args.save_debug)
