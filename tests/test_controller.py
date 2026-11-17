from minichess.controller import GameController
from minichess.game import QUEEN, make_piece, BLACK, WHITE, State, initial_state, parse_square as sq


def board(diagram: str, turn=WHITE) -> State:
    rows = diagram.split()
    return State.from_string("".join(reversed(rows)) + "wb"[turn])


def test_targets_only_for_movable_own_pieces():
    c = GameController()
    assert sorted(c.targets_from(sq("b1"))) == sorted([sq("a3"), sq("c3")])
    assert c.targets_from(sq("a2")) == [sq("a3")]
    assert c.targets_from(sq("a1")) == []   # blocked rook
    assert c.targets_from(sq("a4")) == []   # opponent's piece
    assert c.targets_from(sq("c3")) == []   # empty square


def test_try_move_legal_illegal_and_turn_switch():
    c = GameController()
    assert not c.try_move(sq("a2"), sq("a4"))
    assert c.state == initial_state()
    assert c.try_move(sq("a2"), sq("a3"))
    assert c.state.turn == BLACK and c.state.ply == 1
    assert c.last_action is not None
    assert not c.try_move(sq("b2"), sq("b3"))   # White piece, Black's turn


def test_undo_and_reset():
    c = GameController()
    c.try_move(sq("a2"), sq("a3"))
    c.try_move(sq("a4"), sq("a3")) or c.try_move(sq("b4"), sq("b3"))
    assert c.undo() and c.undo()
    assert c.state == initial_state() and c.last_action is None
    assert not c.undo()
    c.try_move(sq("a2"), sq("a3"))
    c.reset()
    assert c.state == initial_state()


def test_game_over_locks_input():
    mate = board("""
        k....
        .Q...
        ..K..
        .....
        .....
    """, turn=BLACK)
    c = GameController(mate)
    assert c.game_over and c.result == 1 and c.reason == "checkmate"
    assert c.targets_from(sq("a5")) == []
    assert not c.try_move(sq("a5"), sq("a4"))


def test_pawn_promotes_and_game_continues():
    s = board("""
        .....
        .P...
        ....k
        .....
        K....
    """)
    c = GameController(s)
    assert not c.game_over
    assert c.try_move(sq("b4"), sq("b5"))
    assert not c.game_over
    assert c.state.squares[sq("b5")] == make_piece(QUEEN, WHITE)
    assert c.undo() and not c.game_over
