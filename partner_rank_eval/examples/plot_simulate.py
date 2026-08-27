#!/usr/bin/env python3
"""Two-panel identity-grid + gallery-nesting figure (Supplementary Figure S4)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from partner_rank_eval.simulate import (
    DEFAULT_NESTED_GALLERY_SIZES,
    simulate_gallery_nesting,
    simulate_grid,
)

HERE = Path(__file__).resolve().parent
OUT_LOCAL = HERE / "figures"
BIB_ROOT = HERE.parents[1]  # bib_0802 (examples/ -> partner_rank_eval/ -> bib_0802/)
OUT_DIRS = [
    OUT_LOCAL,
    BIB_ROOT / "01_manuscript_submit" / "05_Supplementary_Figures",
    BIB_ROOT / "transfer_bioinformatics" / "05_Supplementary_Figures",
]
NAVY = "#1f4e79"
ORANGE = "#c45911"
# Empirical HVIDB pair-subset saturation (P(|G_q| ≤ 10) = 85.1%)
HVIDB_SATURATION_P = 0.851
K_HIT = 10


def main() -> None:
    nested = DEFAULT_NESTED_GALLERY_SIZES
    grid = simulate_grid(n_queries=4000, seed=0)
    nest = simulate_gallery_nesting(
        n_queries=4000, seed=0, nested_sizes=nested, full_gallery=max(nested)
    )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))

    ax = axes[0]
    for q, color in ((0.5, NAVY), (0.9, ORANGE)):
        xs, ys, pred = [], [], []
        for row in grid:
            if abs(row["design_oracle@K"] - q) < 1e-9:
                xs.append(row["design_recall@T"])
                ys.append(row["end_to_end_hit@K"])
                pred.append(row["design_hit@K"])
        order = np.argsort(xs)
        xs_a = np.array(xs)[order]
        pred_a = np.array(pred)[order]
        ys_a = np.array(ys)[order]
        ax.plot(xs_a, pred_a, "--", color=color, lw=1.2, label=f"r × q (q={q})")
        ax.scatter(xs_a, ys_a, s=28, color=color, zorder=3, label=f"simulated Hit@10 (q={q})")
    ax.set_xlabel("Designed retrieval Recall@T (r)")
    ax.set_ylabel("End-to-end Hit@K")
    ax.set_title("A  Hit@K = Recall@T × oracle@K", loc="left", fontsize=9, fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")

    ax = axes[1]
    sizes = [r["gallery_size"] for r in nest]
    hits = [r["end_to_end_hit@K"] for r in nest]
    ax.plot(sizes, hits, "o-", color=NAVY, lw=1.2, ms=5, label="Fixed ranking quality")
    # Structural Hit@K = 1 band for |G| ≤ K
    ax.axvspan(0.8, K_HIT, color=ORANGE, alpha=0.12, zorder=0)
    ax.axvline(K_HIT, color=ORANGE, ls=":", lw=1.0)
    ax.annotate(
        f"|G|≤{K_HIT}: Hit@{K_HIT}=1\n(by construction)",
        xy=(K_HIT, 1.0),
        xytext=(35, 0.78),
        fontsize=7,
        color=ORANGE,
        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.8),
    )
    ax.annotate(
        f"HVIDB pair-subset:\nP(|G_q|≤10)={100*HVIDB_SATURATION_P:.1f}%",
        xy=(2, hits[-1] if hits else 1.0),
        xytext=(8, 0.35),
        fontsize=7,
        color=NAVY,
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=0.8),
    )
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks(list(sizes))
    ax.set_xticklabels([str(s) for s in sizes], fontsize=7)
    ax.set_xlabel("Evaluation gallery size |G| (log; large/fixed → small/pair-subset)")
    ax.set_ylabel("Headline Hit@K")
    ax.set_title(
        "B  Hit@K vs gallery size (fixed scores)",
        loc="left",
        fontsize=9,
        fontweight="bold",
    )
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")

    fig.tight_layout(w_pad=2.0)
    summary = {
        "n_queries": 4000,
        "seed": 0,
        "T": 20,
        "K": 10,
        "recalls": [0.05, 0.2, 0.5, 0.8],
        "oracles": [0.5, 0.9],
        "identity_max_abs_error": max(r["identity_abs_error"] for r in grid),
        "identity_grid": [
            {
                "r": r["design_recall@T"],
                "q": r["design_oracle@K"],
                "hit@K": r["end_to_end_hit@K"],
                "r_times_q": r["design_hit@K"],
                "identity_abs_error": r["identity_abs_error"],
            }
            for r in grid
        ],
        "nesting_sizes": sizes,
        "nesting_hits": hits,
        "nesting_curve": [
            {
                "gallery_size": r["gallery_size"],
                "hit@K": r["end_to_end_hit@K"],
                "protocol": r.get("protocol"),
                "structural_hitK_unity": r["gallery_size"] <= K_HIT,
            }
            for r in nest
        ],
        "hvidb_pair_subset_P_G_le_10": HVIDB_SATURATION_P,
        "inflation_vs_full_gallery": nest[-1].get("inflation_vs_full_gallery"),
        "monotonic_nondecreasing_hit": nest[-1].get("monotonic_nondecreasing_hit"),
        "note": (
            "Panel B uses fixed underlying scores with nested evaluation galleries "
            f"|G|∈{list(nested)}; Hit@K=1 whenever |G|≤K and the true partner is in-gallery."
        ),
    }
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "FigureS4_identity_nesting.png", dpi=300)
        fig.savefig(out_dir / "FigureS4_identity_nesting.pdf")
        fig.savefig(out_dir / "FigureS4.png", dpi=300)
        (out_dir / "FigureS4_simulate_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
    plt.close(fig)
    (OUT_LOCAL / "simulate_summary.json").write_text(json.dumps(summary, indent=2))
    # also drop a copy under transfer analysis/
    analysis = BIB_ROOT / "transfer_bioinformatics" / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    (analysis / "gallery_size_hitk_curve.json").write_text(json.dumps(summary, indent=2))

    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "identity_max_abs_error",
                    "nesting_sizes",
                    "nesting_hits",
                    "inflation_vs_full_gallery",
                    "monotonic_nondecreasing_hit",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
