"""Draw a mini chess position with matplotlib: onto any Axes, or to a PNG file.

No pyplot / global backend is touched here, so this works both for headless PNG export
and inside the interactive drag-and-drop window (``minichess.gui``).
"""
from __future__ import annotations

from typing import Iterable, Optional

import matplotlib.patheffects as pe
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle

from .game import (
    EMPTY, KING, SIZE, State, color_of, decode_move, in_check, kind_of, make_piece, status,
)

LIGHT, DARK = "#f0d9b5", "#b58863"
LAST_MOVE = "#f7ec74"
SELECTED = "#bbcb44"
CHECK_RED = "#e5534b"
# Solid glyphs for both sides; the fill colour tells them apart.
GLYPHS = {1: "♟", 2: "♞", 3: "♝", 4: "♜", 5: "♛", 6: "♚"}
SIDE = ("White", "Black")


def piece_artist(ax, piece: int, x: float, y: float, animated: bool = False):
    """Add a piece glyph centred at data coords (x, y). Returns the Text artist."""
    white = color_of(piece) == 0
    return ax.text(
        x, y, GLYPHS[kind_of(piece)],
        ha="center", va="center", fontsize=46, family="DejaVu Sans", zorder=5,
        color="#fdfdfd" if white else "#1b1b1b", animated=animated,
        path_effects=[pe.withStroke(linewidth=2.4, foreground="#1b1b1b" if white else "#fdfdfd")],
    )


def paint_position(
    ax,
    state: State,
    last_action: Optional[int] = None,
    hidden_sq: Optional[int] = None,
    selected_sq: Optional[int] = None,
    targets: Iterable[int] = (),
) -> None:
    """(Re)draw the whole board on ``ax``.

    hidden_sq: square whose piece is not drawn (it is being dragged).
    selected_sq: square tinted as the picked-up piece's origin.
    targets: squares to mark as legal destinations (dot = move, ring = capture).
    """
    ax.clear()
    ax.set_xlim(0, SIZE)
    ax.set_ylim(0, SIZE)
    ax.set_aspect("equal")

    last = set(decode_move(last_action)) if last_action is not None else set()
    checked = state.squares.index(make_piece(KING, state.turn)) if in_check(state) else None

    for sq in range(SIZE * SIZE):
        r, f = divmod(sq, SIZE)
        ax.add_patch(Rectangle((f, r), 1, 1, facecolor=LIGHT if (r + f) % 2 else DARK,
                               edgecolor="none", zorder=0))
        for flag, color, alpha in ((sq in last, LAST_MOVE, 0.6), (sq == selected_sq, SELECTED, 0.75),
                                   (sq == checked, CHECK_RED, 0.65)):
            if flag:
                ax.add_patch(Rectangle((f, r), 1, 1, facecolor=color, alpha=alpha,
                                       edgecolor="none", zorder=1))
        piece = state.squares[sq]
        if piece != EMPTY and sq != hidden_sq:
            piece_artist(ax, piece, f + 0.5, r + 0.47)

    for sq in targets:
        r, f = divmod(sq, SIZE)
        if state.squares[sq] == EMPTY:
            ax.add_patch(Circle((f + 0.5, r + 0.5), 0.15, facecolor="#222", alpha=0.3,
                                edgecolor="none", zorder=2))
        else:
            ax.add_patch(Circle((f + 0.5, r + 0.5), 0.45, fill=False, edgecolor="#222",
                                linewidth=4, alpha=0.35, zorder=2))

    for i in range(SIZE):
        ax.text(i + 0.5, -0.18, "abcde"[i], ha="center", va="center", fontsize=13, color="#444")
        ax.text(-0.18, i + 0.5, str(i + 1), ha="center", va="center", fontsize=13, color="#444")
    ax.axis("off")

    result, reason = status(state)
    if result is not None:
        title = {1: "White wins", -1: "Black wins", 0: "Draw"}[result] + f" ({reason})"
    else:
        title = f"{SIDE[state.turn]} to move" + ("  –  CHECK" if checked is not None else "")
    ax.set_title(title, fontsize=15, pad=14, color="#222")


def draw_board(state: State, path: str, last_action: Optional[int] = None) -> None:
    """Save the position as a PNG (headless, no window)."""
    fig = Figure(figsize=(6, 6.6), dpi=110)
    FigureCanvasAgg(fig)
    paint_position(fig.add_subplot(), state, last_action)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
