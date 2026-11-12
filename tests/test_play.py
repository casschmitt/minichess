from minichess.game import BLACK, WHITE, State, initial_state, parse_square
from minichess.play import explain_illegal, parse_move, play, render


def board(diagram: str, turn=WHITE) -> State:
    rows = diagram.split()
    return State.from_string("".join(reversed(rows)) + "wb"[turn])


def run(moves, state=None):
    """Feed scripted input to play(); return (result, printed output)."""
    feed = iter(moves)
    out = []

    def fake_input(prompt):
        out.append(prompt)
        try:
            return next(feed)
        except StopIteration:
            raise EOFError

    result = play(state, input_fn=fake_input, output_fn=out.append)
    return result, "\n".join(out)


def test_render_initial_board():
    text = render(initial_state())
    assert text.splitlines()[0] == "5  r n b q k"
    assert text.splitlines()[1] == "4  p p p p p"
    assert text.splitlines()[4] == "1  R N B Q K"
    assert text.splitlines()[-1] == "   a b c d e"


def test_parse_move_formats():
    expected = (parse_square("a2"), parse_square("a3"))
    assert parse_move("a2a3") == expected
    assert parse_move("a2 a3") == expected
    assert parse_move(" A2-A3 ") == expected
    assert parse_move("a2a6") is None
    assert parse_move("hello") is None
    assert parse_move("") is None


def test_illegal_move_reasons():
    s = initial_state()
    assert "no piece" in explain_illegal(s, parse_square("c3"), parse_square("c4"))
    assert "opponent" in explain_illegal(s, parse_square("a4"), parse_square("a3"))
    assert "your own piece" in explain_illegal(s, parse_square("a1"), parse_square("a2"))
    assert "pawn" in explain_illegal(s, parse_square("a2"), parse_square("a4"))
    assert "cannot move" in explain_illegal(s, parse_square("b1"), parse_square("b3"))
    pinned = board("""
        r...k
        .....
        B....
        .....
        K....
    """)
    assert "check" in explain_illegal(pinned, parse_square("a3"), parse_square("b4"))


def test_scripted_game_with_rejections_ends_in_pawn_win():
    s = board("""
        .....
        .P...
        ....k
        .....
        K....
    """)
    result, out = run(["xx", "e3e4", "a1a3", "b4b5"], s)
    assert result == 1
    assert "Could not read that" in out
    assert "belongs to your opponent" in out
    assert "cannot move" in out
    assert "Game over: White wins (white pawn reached the last rank)." in out


def test_checkmate_is_announced():
    s = board("""
        k....
        .Q...
        ..K..
        .....
        .....
    """, turn=BLACK)
    result, out = run([], s)
    assert result == 1
    assert "Game over: White wins (checkmate)." in out


def test_check_is_announced_and_quit_works():
    s = board("""
        k....
        .Q...
        ..K..
        .....
        .....
    """, turn=WHITE)
    result, out = run(["b4b5", "quit"], s)  # Qb5+ attacks the black king on a5
    assert result is None
    assert "Check!" in out
    assert "Game abandoned." in out


def test_turns_alternate_and_prompt_names_side():
    result, out = run(["a2a3", "a4a3"])  # second move is Black pawn pushing into blocked square
    assert "White to move" in out
    assert "Black to move" in out
    assert "Illegal move" in out  # a4a3 is blocked by the white pawn on a3 -> rejected
    assert result is None  # input ran out (EOF) -> abandoned
