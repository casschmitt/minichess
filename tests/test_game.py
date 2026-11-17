import random

from minichess.game import (
    BLACK, KING, KNIGHT, PAWN, QUEEN, WHITE, State, apply, decode_move, encode_move, in_check,
    initial_state, is_attacked, legal_moves, make_piece, move_name,
    parse_square, pseudo_legal_moves, status,
)


def board(diagram: str, turn=WHITE) -> State:
    """Build a State from 5 rows of 5 chars, rank 5 on top (as printed on screen)."""
    rows = diagram.split()
    assert len(rows) == 5 and all(len(r) == 5 for r in rows)
    return State.from_string("".join(reversed(rows)) + "wb"[turn])


def names(state):
    return sorted(move_name(a) for a in legal_moves(state))


# ---------------------------------------------------------------- initial position

def test_initial_layout():
    s = initial_state()
    assert s.to_string() == "RNQKNPPPPP" + "." * 5 + "ppppp" + "rnqkn" + "w"
    assert s.turn == WHITE and s.ply == 0


def test_initial_legal_moves_white_and_black():
    s = initial_state()
    assert names(s) == sorted(["a2a3", "b2b3", "c2c3", "d2d3", "e2e3", "b1a3", "b1c3", "e1d3"])
    black = State(s.squares, BLACK)
    assert names(black) == sorted(["a4a3", "b4b3", "c4c3", "d4d3", "e4e3", "b5a3", "b5c3", "e5d3"])


def test_apply_pawn_push():
    s = apply(initial_state(), encode_move(parse_square("a2"), parse_square("a3")))
    assert s.squares[parse_square("a2")] == 0
    assert s.squares[parse_square("a3")] == 1  # white pawn
    assert s.turn == BLACK and s.ply == 1


# ---------------------------------------------------------------- serialization

def test_pack_roundtrip_and_size():
    s = initial_state()
    data = s.pack()
    assert len(data) == 13
    back = State.unpack(data)
    assert back.squares == s.squares and back.turn == s.turn


def test_string_roundtrip():
    s = initial_state()
    assert State.from_string(s.to_string()) == s


# ---------------------------------------------------------------- movement details

def test_pawn_blocked_push_and_diagonal_capture_only():
    s = board("""
        ....k
        .bn..
        ..P..
        .....
        K....
    """)
    pawn_sq = parse_square("c3")
    moves = [move_name(a) for a in legal_moves(s) if decode_move(a)[0] == pawn_sq]
    assert moves == ["c3b4"]  # c4 is blocked, d4 is empty so no capture there


def test_pinned_piece_cannot_move():
    s = board("""
        r...k
        .....
        B....
        .....
        K....
    """)
    bishop_sq = parse_square("a3")
    assert any(f == bishop_sq for f, _ in pseudo_legal_moves(s))  # it can move on paper
    assert all(decode_move(a)[0] != bishop_sq for a in legal_moves(s))  # but not legally


def test_slider_attack_is_blocked_by_first_piece():
    s = board("""
        ....k
        .....
        .....
        .....
        RPr.K
    """)
    # black rook on c1 attacks west along rank 1, but the white pawn on b1 blocks the ray
    assert is_attacked(s.squares, parse_square("b1"), BLACK)      # rook c1 hits b1
    assert not is_attacked(s.squares, parse_square("a1"), BLACK)  # shielded by the pawn on b1


# ---------------------------------------------------------------- game end

def test_checkmate():
    s = board("""
        k....
        .Q...
        ..K..
        .....
        .....
    """, turn=BLACK)
    assert in_check(s)
    assert legal_moves(s) == []
    assert status(s) == (1, "checkmate")


def test_stalemate():
    s = board("""
        k....
        ..Q..
        .....
        .....
        ....K
    """, turn=BLACK)
    assert not in_check(s)
    assert legal_moves(s) == []
    assert status(s) == (0, "stalemate")


def test_white_pawn_promotes_to_queen():
    s = board("""
        .....
        .P...
        ....k
        .....
        K....
    """)
    assert status(s)[0] is None
    s2 = apply(s, encode_move(parse_square("b4"), parse_square("b5")))
    assert s2.squares[parse_square("b5")] == make_piece(QUEEN, WHITE)
    assert status(s2)[0] is None


def test_black_pawn_promotes_to_queen():
    s = board("""
        k...K
        .....
        .....
        ..p..
        .....
    """, turn=BLACK)
    s2 = apply(s, encode_move(parse_square("c2"), parse_square("c1")))
    assert s2.squares[parse_square("c1")] == make_piece(QUEEN, BLACK)
    assert status(s2)[0] is None


def test_insufficient_material_draws():
    kk = board("""
        k....
        .....
        .....
        .....
        ....K
    """)
    assert status(kk) == (0, "insufficient material")
    kn = board("""
        k....
        .....
        .N...
        .....
        ....K
    """)
    assert status(kn)[0] == 0
    kr = board("""
        k....
        .....
        .R...
        .....
        ....K
    """)
    assert status(kr)[0] is None


def test_ply_cap_draw():
    s = initial_state()
    assert status(s)[0] is None
    assert status(State(s.squares, s.turn, 150)) == (0, "move limit reached")


# ---------------------------------------------------------------- random playouts

def test_random_playouts_keep_invariants():
    rng = random.Random(0)
    finished = 0
    for _ in range(200):
        s = initial_state()
        while True:
            result, _ = status(s)
            if result is not None:
                finished += 1
                break
            assert State.unpack(s.pack()).squares == s.squares
            assert State.from_string(s.to_string()).squares == s.squares
            moves = legal_moves(s)
            assert moves
            s = apply(s, rng.choice(moves))
            # the side that just moved must never be left in check
            mover = 1 - s.turn
            king_sq = s.squares.index(make_piece(KING, mover))
            assert not is_attacked(s.squares, king_sq, s.turn)
    assert finished == 200


def test_pawn_double_step():
    start = initial_state()
    # pawns start on ranks 2/4, so the double step is blocked in the initial position
    assert "a2a4" not in names(start)
    sq = list(start.squares)
    sq[parse_square("a4")] = 0
    assert {"a2a3", "a2a4"} <= set(names(State(tuple(sq), WHITE)))
    # blocked intermediate square
    sq[parse_square("a3")] = make_piece(KNIGHT, BLACK)
    assert "a2a4" not in names(State(tuple(sq), WHITE))
    # no double step from a non-start rank
    sq[parse_square("a3")] = make_piece(PAWN, WHITE)
    sq[parse_square("a2")] = 0
    assert "a3a5" not in names(State(tuple(sq), WHITE))
    # black mirror: a4 pawn, a3/a2 empty
    sq = list(start.squares)
    sq[parse_square("a2")] = 0
    assert {"a4a3", "a4a2"} <= set(names(State(tuple(sq), BLACK)))
