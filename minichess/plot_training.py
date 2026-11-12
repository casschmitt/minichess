"""Plot training progress from the log written by ``minichess.train``.

    python -m minichess.plot_training      # checkpoints/log.jsonl -> plots/training.png

Three panels (no dual axis; the two losses live on different scales so each gets its own):
policy loss, value loss, and playing strength (score vs a random mover, and vs the previous
best net). Score = wins + half the draws, over games played.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3, in order


def load_log(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def final_eval_lines(log_path) -> list[str]:
    """The final-evaluation lines printed by train.py, if train.log sits next to the log."""
    train_log = Path(log_path).with_name("train.log")
    if not train_log.exists():
        return []
    keep = ("vs random mover:", "vs untrained net:")
    # "vs random mover: 35W 3D 2L (score 0.91; target >= 0.90)" -> drop the target part
    return [re.sub(r";[^)]*\)", ")", line)
            for line in train_log.read_text().splitlines() if line.startswith(keep)]


def trailing_mean(values, window: int = 5) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    return np.array([v[max(0, i - window + 1): i + 1].mean() for i in range(len(v))])


def style(ax, n_iters: int) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(color=AXIS, labelcolor=INK2, labelsize=10, length=3, width=1)
    ax.grid(axis="y", color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    step = 1 if n_iters <= 12 else 5 if n_iters <= 40 else 10   # readable ticks for any run length
    ax.set_xticks([0] + list(range(step, n_iters + 1, step)))
    ax.set_xlabel("training iteration", color=MUTED, fontsize=10)


def panel_title(ax, title: str, subtitle: str | None = None) -> None:
    ax.annotate(title, (0, 1), xycoords="axes fraction", xytext=(0, 31 if subtitle else 30),
                textcoords="offset points", fontsize=12.5, weight="bold", color=INK, va="bottom")
    if subtitle:
        ax.annotate(subtitle, (0, 1), xycoords="axes fraction", xytext=(0, 12),
                    textcoords="offset points", fontsize=9.5, color=MUTED, va="bottom")


def end_dot(ax, x, y, color) -> None:
    """Marker with a 2px surface ring so it stays legible over the line."""
    ax.plot([x], [y], marker="o", markersize=8, color=color, markeredgecolor=SURFACE,
            markeredgewidth=2, linestyle="none", zorder=5, clip_on=False)


def tag(ax, x, y, text, dx=9, dy=0, ha="left", va="center") -> None:
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                fontsize=10.5, color=INK)


def loss_panel(ax, iters, values, title, subtitle, mark_min=False) -> None:
    style(ax, iters[-1])
    ax.plot(iters, values, color=BLUE, linewidth=2, solid_capstyle="round", solid_joinstyle="round")
    ax.margins(y=0.24 if mark_min else 0.14)   # room for the "lowest" label under the line
    ax.set_xlim(0, iters[-1] + 6)
    tag(ax, iters[0], values[0], f"{values[0]:.2f}", dy=6)
    if mark_min:
        i = int(np.argmin(values))
        end_dot(ax, iters[i], values[i], BLUE)
        tag(ax, iters[i], values[i], f"lowest {values[i]:.2f} (iteration {iters[i]})", dy=-14)
    end_dot(ax, iters[-1], values[-1], BLUE)
    tag(ax, iters[-1], values[-1], f"{values[-1]:.2f}", dy=9 if mark_min else 0)
    panel_title(ax, title, subtitle)


def strength_panel(ax, rows) -> None:
    iters = [r["iter"] for r in rows]
    style(ax, iters[-1])
    vs_random = [r["vs_random"] for r in rows]
    smooth = trailing_mean(vs_random)
    best = [(r["iter"], r["vs_best"]) for r in rows if r["vs_best"] is not None]

    ax.axhline(0.5, color=AXIS, linewidth=1, zorder=1)
    ax.annotate("0.5 = even with the previous best net", (1, 0.5), xycoords=("axes fraction", "data"),
                xytext=(-4, -13), textcoords="offset points", fontsize=9.5, color=MUTED,
                ha="right", va="top")
    ax.plot(iters, vs_random, color=ORANGE, linewidth=1.5, alpha=0.35)
    ax.plot(iters, smooth, color=ORANGE, linewidth=2, solid_capstyle="round")
    end_dot(ax, iters[-1], smooth[-1], ORANGE)
    tag(ax, iters[-1], smooth[-1], f"{smooth[-1]:.2f}")
    if best:
        bx, by = zip(*best)
        ax.plot(bx, by, color=AQUA, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, solid_capstyle="round")
        tag(ax, bx[-1], by[-1], f"{by[-1]:.2f}")

    ax.set_xlim(0, iters[-1] + 6)
    ax.set_ylim(0.4, 1.03)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    handles = [Line2D([0], [0], color=ORANGE, linewidth=2),
               Line2D([0], [0], color=AQUA, linewidth=2, marker="o", markersize=8,
                      markeredgecolor=SURFACE, markeredgewidth=2)]
    labels = ["vs random mover (faint: per iteration · bold: 5-iteration average)",
              "vs previous best net (every 5 iterations)"]
    legend = ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
                       frameon=False, fontsize=10, handlelength=1.8, columnspacing=1.6,
                       borderaxespad=0.2)
    for text in legend.get_texts():
        text.set_color(INK2)
    panel_title(ax, "Playing strength")


def make_figure(rows: list[dict], final_lines: list[str]) -> Figure:
    iters = [r["iter"] for r in rows]
    fig = Figure(figsize=(12, 8.2), dpi=150, facecolor=SURFACE)
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(2, 2, left=0.065, right=0.96, top=0.815, bottom=0.165,
                          hspace=0.72, wspace=0.17, height_ratios=[1, 1.1])
    loss_panel(fig.add_subplot(gs[0, 0]), iters, [r["policy_loss"] for r in rows], "Policy loss",
               "cross-entropy against the search's move distribution (lower is better)")
    loss_panel(fig.add_subplot(gs[0, 1]), iters, [r["value_loss"] for r in rows], "Value loss",
               "squared error against the final game result (lower is better)", mark_min=True)
    strength_panel(fig.add_subplot(gs[1, :]), rows)

    fig.text(0.065, 0.945, "Mini chess self-play training", fontsize=19, weight="bold", color=INK)
    fig.text(0.065, 0.905, f"{iters[-1]} iterations in {rows[-1]['elapsed_min']:.0f} minutes · "
             f"{rows[-1]['games']} self-play games per iteration", fontsize=11.5, color=INK2)
    if final_lines:
        fig.text(0.065, 0.065, "Final test:  " + "   ·   ".join(final_lines), fontsize=10,
                 color=INK2)
    fig.text(0.065, 0.034, "Score = wins + ½ draws, over games played. Per-iteration points use "
             "only 16 games (20 for the previous-best match), so they are noisy.",
             fontsize=9.5, color=MUTED)
    return fig


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Plot training progress.")
    parser.add_argument("--log", default="checkpoints/log.jsonl")
    parser.add_argument("--out", default="plots/training.png")
    args = parser.parse_args(argv)
    rows = load_log(args.log)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    make_figure(rows, final_eval_lines(args.log)).savefig(out, facecolor=SURFACE)
    print(f"wrote {out} ({len(rows)} iterations)")


if __name__ == "__main__":
    main()
