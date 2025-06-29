#!/usr/bin/env python3
# solve_puzzle_borders.py
#
# "Greedy" assembler based on contours. Exports the
# `solve_greedy()` function for other modules and can still be
# used as a script. When run from the CLI, it shows each piece’s
# rotation (0 / 90 / 180 / 270 degrees) applied by the solver.
# ------------------------------------------------------------------------------

from __future__ import annotations
import os
import itertools
from collections import defaultdict
from glob import glob
from math import sqrt
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon
from numpy.fft import fft

# ────────── CONSTANTS AND MAPS ──────────
STRIPE_SAMPLES   = 100
MAX_DEVIATION_PX = 3.0
SIDES = ["top", "right", "bottom", "left"]
ROT_MAP = {
    0:   SIDES,
    90:  ["left", "top", "right", "bottom"],
    180: ["bottom", "left", "top", "right"],
    270: ["right", "bottom", "left", "top"],
}
OPPOSITE   = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
DIR_OFFSET = {"top": (-1, 0), "right": (0, 1), "bottom": (1, 0), "left": (0, -1)}

# ─────────── UTILITIES ───────────
def rotate_image(img: np.ndarray, rot: int) -> np.ndarray:
    return np.rot90(img, k=rot // 90)

def original_side_for(rot: int, global_side: str) -> str:
    return SIDES[ROT_MAP[rot].index(global_side)]

# ────────── LOADING & EXTRACTION ──────────
def load_piece_contours(folder: str):
    pieces = []
    for p in sorted(glob(os.path.join(folder, "*.png"))):
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        if img.shape[2] == 4:
            alpha = img[:, :, 3]
            mask = (alpha > 0).astype(np.uint8) * 255
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        cnt = cnts[0][:, 0, :]
        pieces.append((Path(p).name, Polygon(cnt), cnt, img))
    return pieces


def detect_max_distance_corners(contour: np.ndarray) -> np.ndarray:
    eps    = 0.01 * cv2.arcLength(contour.reshape(-1, 1, 2).astype(np.int32), True)
    approx = cv2.approxPolyDP(contour.reshape(-1, 1, 2).astype(np.int32), eps, True)[:, 0, :]
    best, maxd = None, -1.0
    for combo in itertools.combinations(approx, 4):
        d = sum(np.linalg.norm(p1 - p2) for p1, p2 in itertools.combinations(combo, 2))
        if d > maxd:
            maxd, best = d, combo
    return np.array(best, np.float32) if best is not None else np.empty((0, 2), np.float32)


def extract_borders_from_contour(contour: np.ndarray):
    pts = contour.reshape(-1, 2)
    corners = detect_max_distance_corners(pts)

    if len(corners) != 4:
        return {s: LineString([]) for s in SIDES}, corners

    center  = corners.mean(axis=0)
    corners = sorted(corners, key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
    names   = ["left", "bottom", "right", "top"]
    idxs    = sorted(np.argmin(np.linalg.norm(pts - c, axis=1)) for c in corners)

    segments = []
    for i in range(4):
        i0, i1 = idxs[i], idxs[(i + 1) % 4]
        seg    = pts[i0:i1 + 1] if i1 >= i0 else np.vstack([pts[i0:], pts[:i1 + 1]])
        segments.append(seg)

    return {names[i]: LineString(segments[i]) for i in range(4)}, corners


def is_straight(border: LineString, max_dev: float = MAX_DEVIATION_PX) -> bool:
    if border.is_empty or len(border.coords) < 2:
        return False
    p0, p1 = map(np.asarray, [border.coords[0], border.coords[-1]])
    v, L = p1 - p0, np.linalg.norm(p1 - p0)
    if L < 1e-3:
        return False
    u = v / L
    return all(
        np.linalg.norm(np.asarray(p) - p0 - np.dot(np.asarray(p) - p0, u) * u) <= max_dev
        for p in border.coords[1:-1]
    )


def classify_straight_sides(borders: dict) -> List[str]:
    return [s for s, br in borders.items() if is_straight(br)]

# ─────────── SIMILARITY METRICS ───────────
def compute_curvature(coords: np.ndarray, window: int = 5) -> np.ndarray:
    n = len(coords)
    angs = []
    for i in range(window, n - window):
        p1, p2, p3 = coords[i - window], coords[i], coords[i + window]
        v1, v2     = p2 - p1, p3 - p2
        n1, n2     = np.linalg.norm(v1), np.linalg.norm(v2)
        ang = 0.0 if n1 < 1e-6 or n2 < 1e-6 else np.arccos(
            np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        angs.append(ang)
    return np.array(angs, np.float32)


def fourier_descriptors(coords: np.ndarray, num: int = 20) -> np.ndarray:
    c = coords[:, 0] + 1j * coords[:, 1]
    F = fft(c)
    mags = np.abs(F[1:num + 1])
    return mags / mags[0] if mags.size and mags[0] > 0 else mags


def resample_border(coords: np.ndarray, num: int = STRIPE_SAMPLES) -> np.ndarray:
    line = LineString(coords)
    ds = np.linspace(0, line.length, num)
    return np.array([line.interpolate(d).coords[0] for d in ds], np.float32)


def compare_borders(b1: LineString, b2: LineString) -> Tuple[float, float]:
    c1, c2 = np.array(b1.coords), np.array(b2.coords)[::-1]
    r1, r2 = resample_border(c1), resample_border(c2)
    cu1, cu2 = compute_curvature(r1), compute_curvature(r2)
    fd1, fd2 = fourier_descriptors(r1), fourier_descriptors(r2)
    return float(np.linalg.norm(cu1 - cu2[:len(cu1)])), float(np.linalg.norm(fd1 - fd2))

# ─────────── PLACEMENT ───────────
def build_allowed_positions(cache: dict, side: int) -> dict:
    allowed = {}
    for idx, info in cache.items():
        straight0 = info["straight"]
        perms = set()

        for rot in (0, 90, 180, 270):
            g = {ROT_MAP[rot][SIDES.index(s)] for s in straight0}

            for r in range(side):
                for c in range(side):
                    corner = (r in (0, side - 1) and c in (0, side - 1))
                    border = (not corner and (r in (0, side - 1) or c in (0, side - 1)))

                    if corner:
                        req = {"top" if r == 0 else "bottom",
                               "left" if c == 0 else "right"}
                        if len(g) == 2 and g == req:
                            perms.add((r, c, rot))

                    elif border:
                        if len(g) == 1:
                            req = ("top"    if r == 0 else
                                   "bottom" if r == side - 1 else
                                   "left"   if c == 0 else "right")
                            if next(iter(g)) == req:
                                perms.add((r, c, rot))
                    else:  # interior
                        if len(g) == 0:
                            perms.add((r, c, rot))
        allowed[idx] = perms
    return allowed


def solver_greedy(cache: dict) -> Tuple[dict, float]:
    n, side = len(cache), int(round(sqrt(len(cache))))
    allowed = build_allowed_positions(cache, side)

    neighbours = {
        (r, c): [(r + dr, c + dc, sd) for sd, (dr, dc) in DIR_OFFSET.items()
                 if 0 <= r + dr < side and 0 <= c + dc < side]
        for r in range(side) for c in range(side)
    }

    corners = [i for i, d in cache.items() if len(d["straight"]) == 2]
    start = min(corners, key=lambda i: len(allowed[i]))
    r0, c0, rot0 = sorted(allowed[start])[0]

    places, used, score = {start: (r0, c0, rot0)}, {(r0, c0)}, 0.0

    for r in range(side):
        for c in range(side):
            if (r, c) in used:
                continue

            best = (None, None, float("inf"))
            for p in cache:
                if p in places:
                    continue
                for rot in (0, 90, 180, 270):
                    if (r, c, rot) not in allowed[p]:
                        continue

                    tot = 0.0
                    for rn, cn, sd in neighbours[(r, c)]:
                        if (rn, cn) not in used:
                            continue
                        q = next(idx for idx, pos in places.items() if pos[:2] == (rn, cn))
                        dc, df = compare_borders(
                            cache[p]["borders"][original_side_for(rot, sd)],
                            cache[q]["borders"][original_side_for(places[q][2], OPPOSITE[sd])]
                        )
                        tot += dc + df

                    if tot < best[2]:
                        best = (p, rot, tot)

            if best[0] is None:
                raise RuntimeError(f"No candidate for cell {(r, c)}")

            places[best[0]] = (r, c, best[1])
            used.add((r, c))
            score += best[2]

    return places, score

# ─────────── FINAL COMPOSITION ───────────
def compose_and_output(cache: dict, places: dict,
                       out: Path, score: float) -> List[List[Tuple[int, int]]]:
    n, side = len(cache), int(round(sqrt(len(cache))))
    matrix = [[None] * side for _ in range(side)]
    for idx, (r, c, rot) in places.items():
        matrix[r][c] = (idx, rot)

    col_w, row_h = defaultdict(int), defaultdict(int)
    for idx, (r, c, rot) in places.items():
        h, w = rotate_image(cache[idx]["img"], rot).shape[:2]
        row_h[r], col_w[c] = max(row_h[r], h), max(col_w[c], w)

    x_offsets = np.concatenate(([0], np.cumsum([col_w[c] for c in range(side - 1)])))
    y_offsets = np.concatenate(([0], np.cumsum([row_h[r] for r in range(side - 1)])))
    canvas = np.full((sum(row_h.values()), sum(col_w.values()), 3), 255, np.uint8)

    for idx, (r, c, rot) in places.items():
        img = rotate_image(cache[idx]["img"], rot)
        y0, x0 = y_offsets[r], x_offsets[c]
        roi = canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]]

        if img.shape[2] == 4:
            alpha = img[:, :, 3:4] / 255.0
            canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = (
                (1 - alpha) * roi + alpha * img[:, :, :3]
            ).astype(np.uint8)
        else:
            canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img[:, :, :3]

    cv2.imwrite(str(out), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"\n🖼️  Saved final assembly → {out}   total_score={score:.4f}")
    return matrix

# ─────────── MAIN API ───────────
def solve_greedy(pieces_dir: str,
                 output_path: str | Path = "solution.png"
                 ) -> Tuple[List[List[Tuple[int, int]]], float, Dict[int, str]]:
    pieces = load_piece_contours(pieces_dir)
    if not pieces:
        raise FileNotFoundError("❌ No PNG files in the input folder.")

    cache: Dict[int, dict] = {}
    for idx, (name, _, cnt, img) in enumerate(pieces):
        borders, _ = extract_borders_from_contour(cnt)
        cache[idx] = {
            "img": img,
            "borders": borders,
            "straight": classify_straight_sides(borders),
            "name": name,
        }
        print(f"✔︎ Piece {idx:02d}: {name}  straight_sides={cache[idx]['straight'] or '—'}")

    places, score = solver_greedy(cache)
    matrix = compose_and_output(cache, places, Path(output_path), score)
    idx2name = {idx: info["name"] for idx, info in cache.items()}
    return matrix, score, idx2name

# ─────────── CLI ───────────
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Greedy contour-based assembler."
    )
    ap.add_argument("-i", "--input",  default="pieces",
                    help="Directory with piece PNGs")
    ap.add_argument("-o", "--output", default="solution_greedy.png",
                    help="Path for the assembled image")
    args = ap.parse_args()

    matrix, score, idx2name = solve_greedy(args.input, args.output)

    print("\nSolver-applied rotations:")
    for row in matrix:
        for idx, rot in row:
            print(f"  • {idx2name[idx]:<25} → {rot:3d}°")

    print("\nPlacement matrix (idx, rot):")
    for row in matrix:
        print(row)
