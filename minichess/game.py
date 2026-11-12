"""5x5 mini chess engine: state, move generation, legality, terminal detection.

Squares are indexed 0..24 as ``rank * 5 + file`` (a1 = 0, e1 = 4, a5 = 20, e5 = 24).
Without castling or en passant, a position is fully described by piece placement
plus the side to move. ``ply`` is only a move counter used for the draw cap.

Rules: pawns push one step and capture diagonally; a pawn reaching the far rank
wins instantly; otherwise standard check / checkmate / stalemate rules apply.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

SIZE = 5
NUM_SQ = SIZE * SIZE
NUM_ACTIONS = NUM_SQ * NUM_SQ  # action = from * 25 + to
MAX_PLY = 150

EMPTY = 0
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(1, 7)
WHITE, BLACK = 0, 1

# Piece codes: 1-6 white (P N B R Q K), 7-12 black. Index into PIECE_CHARS.
PIECE_CHARS = ".PNBRQKpnbrqk"


def make_piece(kind: int, color: int) -> int:
    return kind + 6 * color


def kind_of(piece: int) -> int:
    return (piece - 1) % 6 + 1


def color_of(piece: int) -> int:
    return (piece - 1) // 6


# ---------------------------------------------------------------- geometry tables

def _on(r: int, f: int) -> bool:
    return 0 <= r < SIZE and 0 <= f < SIZE


def _jumps(offsets) -> tuple:
    table = []
    for sq in range(NUM_SQ):
        r, f = divmod(sq, SIZE)
        table.append(tuple(
            (r + dr) * SIZE + (f + df) for dr, df in offsets if _on(r + dr, f + df)
        ))
    return tuple(table)


def _rays(directions) -> tuple:
    table = []
    for sq in range(NUM_SQ):
        r0, f0 = divmod(sq, SIZE)
        rays = []
        for dr, df in directions:
            ray, r, f = [], r0 + dr, f0 + df
            while _on(r, f):
                ray.append(r * SIZE + f)
                r, f = r + dr, f + df
            if ray:
                rays.append(tuple(ray))
        table.append(tuple(rays))
    return tuple(table)


_ORTHO = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))
KNIGHT_TARGETS = _jumps(((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)))
KING_TARGETS = _jumps(_ORTHO + _DIAG)
ORTHO_RAYS = _rays(_ORTHO)
DIAG_RAYS = _rays(_DIAG)


# ---------------------------------------------------------------- squares & moves

def square_name(sq: int) -> str:
    return "abcde"[sq % SIZE] + str(sq // SIZE + 1)


def parse_square(text: str) -> int:
    if len(text) != 2 or text[0] not in "abcde" or text[1] not in "12345":
        raise ValueError(f"bad square {text!r}")
    return (int(text[1]) - 1) * SIZE + "abcde".index(text[0])


def encode_move(frm: int, to: int) -> int:
    return frm * NUM_SQ + to


def decode_move(action: int) -> tuple[int, int]:
    return divmod(action, NUM_SQ)


def move_name(action: int) -> str:
    frm, to = decode_move(action)
    return square_name(frm) + square_name(to)


# ---------------------------------------------------------------- state

@dataclass(frozen=True)
class State:
    squares: tuple  # 25 ints, see PIECE_CHARS
    turn: int = WHITE
    ply: int = 0

    # -- minimal serialization -------------------------------------------------
    def pack(self) -> bytes:
        """25 cells x 4 bits + 1 turn bit = 101 bits -> 13 bytes (ply is not stored)."""
        n = 0
        for i, p in enumerate(self.squares):
            n |= p << (4 * i)
        n |= self.turn << (4 * NUM_SQ)
        return n.to_bytes(13, "little")

    @staticmethod
    def unpack(data: bytes) -> "State":
        n = int.from_bytes(data, "little")
        squares = tuple((n >> (4 * i)) & 0xF for i in range(NUM_SQ))
        return State(squares, (n >> (4 * NUM_SQ)) & 1)

    def to_string(self) -> str:
        """26-char mini-FEN: a1..e5 piece chars, then 'w' or 'b' for the side to move."""
        return "".join(PIECE_CHARS[p] for p in self.squares) + "wb"[self.turn]

    @staticmethod
    def from_string(text: str) -> "State":
        if len(text) != NUM_SQ + 1 or text[-1] not in "wb":
            raise ValueError("expected 25 square chars + 'w'/'b'")
        squares = tuple(PIECE_CHARS.index(c) for c in text[:NUM_SQ])
        return State(squares, "wb".index(text[-1]))


def initial_state() -> State:
    back = (ROOK, KNIGHT, BISHOP, QUEEN, KING)
    squares = [EMPTY] * NUM_SQ
    for f in range(SIZE):
        squares[f] = make_piece(back[f], WHITE)
        squares[SIZE + f] = make_piece(PAWN, WHITE)
        squares[3 * SIZE + f] = make_piece(PAWN, BLACK)
        squares[4 * SIZE + f] = make_piece(back[f], BLACK)
    return State(tuple(squares), WHITE, 0)


# ---------------------------------------------------------------- attacks

def is_attacked(squares, sq: int, by: int) -> bool:
    """True if any piece of colour ``by`` attacks square ``sq``."""
    r, f = divmod(sq, SIZE)
    pr = r - 1 if by == WHITE else r + 1  # rank a pawn of colour `by` would stand on
    if 0 <= pr < SIZE:
        pawn = make_piece(PAWN, by)
        for pf in (f - 1, f + 1):
            if 0 <= pf < SIZE and squares[pr * SIZE + pf] == pawn:
                return True
    knight = make_piece(KNIGHT, by)
    for t in KNIGHT_TARGETS[sq]:
        if squares[t] == knight:
            return True
    king = make_piece(KING, by)
    for t in KING_TARGETS[sq]:
        if squares[t] == king:
            return True
    rook, bishop, queen = make_piece(ROOK, by), make_piece(BISHOP, by), make_piece(QUEEN, by)
    for ray in ORTHO_RAYS[sq]:
        for t in ray:
            p = squares[t]
            if p:
                if p == rook or p == queen:
                    return True
                break
    for ray in DIAG_RAYS[sq]:
        for t in ray:
            p = squares[t]
            if p:
                if p == bishop or p == queen:
                    return True
                break
    return False


def in_check(state: State) -> bool:
    king_sq = state.squares.index(make_piece(KING, state.turn))
    return is_attacked(state.squares, king_sq, 1 - state.turn)


# ---------------------------------------------------------------- move generation

def pseudo_legal_moves(state: State) -> list[tuple[int, int]]:
    """(from, to) pairs obeying piece movement, ignoring whether the king is left in check."""
    squares, turn = state.squares, state.turn
    moves = []
    for sq, p in enumerate(squares):
        if p == EMPTY or color_of(p) != turn:
            continue
        kind = kind_of(p)
        if kind == PAWN:
            r, f = divmod(sq, SIZE)
            nr = r + 1 if turn == WHITE else r - 1
            if 0 <= nr < SIZE:
                if squares[nr * SIZE + f] == EMPTY:
                    moves.append((sq, nr * SIZE + f))
                for nf in (f - 1, f + 1):
                    if 0 <= nf < SIZE:
                        t = nr * SIZE + nf
                        if squares[t] != EMPTY and color_of(squares[t]) != turn:
                            moves.append((sq, t))
        elif kind == KNIGHT or kind == KING:
            for t in (KNIGHT_TARGETS if kind == KNIGHT else KING_TARGETS)[sq]:
                if squares[t] == EMPTY or color_of(squares[t]) != turn:
                    moves.append((sq, t))
        else:
            rays = ()
            if kind in (ROOK, QUEEN):
                rays += ORTHO_RAYS[sq]
            if kind in (BISHOP, QUEEN):
                rays += DIAG_RAYS[sq]
            for ray in rays:
                for t in ray:
                    if squares[t] == EMPTY:
                        moves.append((sq, t))
                    else:
                        if color_of(squares[t]) != turn:
                            moves.append((sq, t))
                        break
    return moves


def _move_squares(squares, frm: int, to: int) -> tuple:
    s = list(squares)
    s[to] = s[frm]
    s[frm] = EMPTY
    return tuple(s)


def legal_moves(state: State) -> list[int]:
    """Legal actions (``from * 25 + to``), sorted. Also the policy mask for the net."""
    turn = state.turn
    king = make_piece(KING, turn)
    result = []
    for frm, to in pseudo_legal_moves(state):
        after = _move_squares(state.squares, frm, to)
        if not is_attacked(after, after.index(king), 1 - turn):
            result.append(encode_move(frm, to))
    result.sort()
    return result


def apply(state: State, action: int) -> State:
    """Play ``action`` (assumed legal) and return the new state."""
    frm, to = decode_move(action)
    return State(_move_squares(state.squares, frm, to), 1 - state.turn, state.ply + 1)


# ---------------------------------------------------------------- terminal detection

def _insufficient_material(squares) -> bool:
    others = [kind_of(p) for p in squares if p != EMPTY and kind_of(p) != KING]
    return not others or (len(others) == 1 and others[0] in (KNIGHT, BISHOP))


def status(state: State, max_ply: int = MAX_PLY,
           legal: Optional[list[int]] = None) -> tuple[Optional[int], str]:
    """(result, reason): result is +1 White wins, -1 Black wins, 0 draw, None if ongoing.

    ``legal`` may carry already-computed ``legal_moves(state)`` to avoid recomputing them.
    """
    sq = state.squares
    if any(sq[(SIZE - 1) * SIZE + f] == make_piece(PAWN, WHITE) for f in range(SIZE)):
        return 1, "white pawn reached the last rank"
    if any(sq[f] == make_piece(PAWN, BLACK) for f in range(SIZE)):
        return -1, "black pawn reached the last rank"
    if legal is None:
        legal = legal_moves(state)
    if not legal:
        if in_check(state):
            return (-1 if state.turn == WHITE else 1), "checkmate"
        return 0, "stalemate"
    if _insufficient_material(sq):
        return 0, "insufficient material"
    if state.ply >= max_ply:
        return 0, "move limit reached"
    return None, ""
