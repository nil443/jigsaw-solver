#!/usr/bin/env python3
"""
main.py – Complete orchestrator for the puzzling pipeline
========================================================

Flow:
    1) Segment → get list of PNGs and centers (initial positions)
    2) Normalize orientation
    3) Solve the puzzle (greedy borders)
    4) Combine angles (normalization + solver)
    5) Save two JSON files:
         • solution_greedy.json   → full result (matrix, score, etc.)
         • pieces_info.json       → only {rotations, initial_positions}

Usage:
    $ python main.py
"""
from __future__ import annotations

from pathlib import Path
import json
from pprint import pformat

from segment_pieces import segment
from normalize_pieces import normalize
from solve_puzzle_borders import solve_greedy

# ─── Paths ───────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
IN_IMG         = BASE_DIR / "in" / "puzzle.png"
SEG_DIR        = BASE_DIR / "pieces"
NORM_DIR       = BASE_DIR / "pieces_normalized"
SOLUTION_PNG   = BASE_DIR / "solution.png"
FULL_JSON_PATH = BASE_DIR / "solution.json"
INFO_JSON_PATH = BASE_DIR / "info_pieces.json"

# Ensure output directories exist
SEG_DIR.mkdir(exist_ok=True)
NORM_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────

def main() -> None:
    print("\n🔄 1. Segmenting pieces")
    png_paths, positions = segment(str(IN_IMG), str(SEG_DIR))
    print("   Centers of pieces:")
    print(pformat(positions, indent=4))

    print("\n🔄 2. Normalizing orientations")
    normalized_rots = normalize(str(SEG_DIR), str(NORM_DIR))

    print("\n🤖 3. Solving puzzle with greedy borders …")
    matrix, score, idx_to_name = solve_greedy(str(NORM_DIR))

    # Combine rotations: normalization + solver
    total_rotations: dict[str, float] = {}
    for row in matrix:
        for idx, solver_angle in row:
            filename = idx_to_name[idx]
            base_angle = normalized_rots.get(filename, 0.0)
            total_rotations[filename] = (base_angle + solver_angle) % 360

    # Prepare info for JSON
    rotations_by_index: dict[int, float] = {
        idx: total_rotations[idx_to_name[idx]] for idx in idx_to_name
    }
    initial_positions: dict[int, tuple[int, int]] = positions

    # Save full solution JSON
    print(f"\n💾  Saving solution data to {FULL_JSON_PATH}")
    with FULL_JSON_PATH.open('w') as f:
        json.dump({
            "matrix": matrix,
            "positions": positions,
            "rotations_normalize": normalized_rots,
            "rotations_total": total_rotations,
            "score": score,
        }, f, indent=2)

    # Save pieces info JSON
    print(f"💾  Saving initial rotations and positions to {INFO_JSON_PATH}")
    with INFO_JSON_PATH.open('w') as f:
        json.dump({
            "rotations": rotations_by_index,
            "initial_positions": initial_positions,
        }, f, indent=2)

    # Final report
    print("\n✅  Process completed.")
    print("   Total rotations per piece:")
    for name in sorted(total_rotations):
        print(f"   • {name:<20} → {total_rotations[name]:6.2f}°")


if __name__ == "__main__":
    main()
