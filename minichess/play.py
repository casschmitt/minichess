"""Two-player command-line mini chess.  Run:  python -m minichess.play"""
from __future__ import annotations

import re
from typing import Callable, Optional

from .game import (
    EMPTY, PAWN, PIECE_CHARS, SIZE, State, apply, color_of, encode_move,
    in_check, initial_state, kind_of, legal_moves, parse_square, pseudo_legal_moves,
    square_name, status,
)

SIDE = ("White", "Black")
MOVE_RE = re.compile(r"^([a-e][1-5])([a-e][1-5])$")
PIECE_NAMES = {1: "pawn", 2: "knight", 3: "bishop", 4: "rook", 5: "queen", 6: "king"}


def render(state: State) -> str:
    """ASCII board, rank 5 at the top. Uppercase = White, lowercase = Black."""
    lines = []
    for r in range(SIZE - 1, -1, -1):
        row = " ".join(PIECE_CHARS[state.squares[r * SIZE + f]] for f in range(SIZE))
        lines.append(f"{r + 1}  {row}")
    lines.append("")
    lines.append("   a b c d e")
    return "\n".join(lines)


def parse_move(text: str) -> Optional[tuple[int, int]]:
    """Accepts 'a2a3', 'a2 a3', 'a2-a3' (any case). Returns (from, to) or None if malformed."""
    m = MOVE_RE.match(re.sub(r"[\s\-]", "", text.lower()))
    return (parse_square(m.group(1)), parse_square(m.group(2))) if m else None


def explain_illegal(state: State, frm: int, to: int) -> str:
    piece = state.squares[frm]
    if frm == to:
        return "the piece has to move to a different square"
    if piece == EMPTY:
        return f"there is no piece on {square_name(frm)}"
    if color_of(piece) != state.turn:
        return f"the piece on {square_name(frm)} belongs to your opponent"
    target = state.squares[to]
    if target != EMPTY and color_of(target) == state.turn:
        return f"{square_name(to)} is occupied by your own piece"
    if (frm, to) in pseudo_legal_moves(state):
        return "that move would leave your king in check"
    kind = PIECE_NAMES[kind_of(piece)]
    if kind_of(piece) == PAWN:
        return "a pawn moves one square forward onto an empty square, or captures diagonally"
    return f"a {kind} cannot move from {square_name(frm)} to {square_name(to)}"


def play(
    state: Optional[State] = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    on_position: Optional[Callable[[State, Optional[int]], None]] = None,
) -> Optional[int]:
    """Run a game. Returns +1 / -1 / 0 for White win / Black win / draw, or None if quit.

    ``on_position(state, last_action)`` is called for every new position (e.g. to draw a PNG).
    """
    state = state or initial_state()
    last_action = None
    while True:
        output_fn("\n" + render(state) + "\n")
        if on_position:
            on_position(state, last_action)
        result, reason = status(state)
        if result is not None:
            winner = {1: "White wins", -1: "Black wins", 0: "Draw"}[result]
            output_fn(f"Game over: {winner} ({reason}).")
            return result
        if in_check(state):
            output_fn("Check!")

        legal = set(legal_moves(state))
        while True:
            try:
                text = input_fn(f"{SIDE[state.turn]} to move (e.g. a2a3, 'quit' to exit): ")
            except EOFError:
                output_fn("\nInput ended. Game abandoned.")
                return None
            if text.strip().lower() in ("quit", "q", "exit"):
                output_fn("Game abandoned.")
                return None
            move = parse_move(text)
            if move is None:
                output_fn("Could not read that. Enter a move like a2a3 (files a-e, ranks 1-5).")
                continue
            action = encode_move(*move)
            if action not in legal:
                output_fn(f"Illegal move: {explain_illegal(state, *move)}.")
                continue
            break
        state = apply(state, action)
        last_action = action


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Two-player 5x5 mini chess.")
    parser.add_argument("--png", default="board.png", metavar="PATH",
                        help="rewrite this PNG after every move (default: board.png)")
    parser.add_argument("--no-png", action="store_true", help="terminal only, no PNG")
    args = parser.parse_args()

    on_position = None
    if not args.no_png:
        from .render_png import draw_board

        def on_position(state, last_action):
            draw_board(state, args.png, last_action)

        print(f"Board image is updated after every move: {args.png}")
    play(on_position=on_position)


if __name__ == "__main__":
    main()
