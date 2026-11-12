from minichess.game import encode_move, initial_state, parse_square
from minichess.play import play
from minichess.render_png import draw_board


def test_draw_board_writes_png(tmp_path):
    out = tmp_path / "board.png"
    draw_board(initial_state(), str(out), encode_move(parse_square("a2"), parse_square("a3")))
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_play_calls_on_position_for_every_position(tmp_path):
    seen = []
    moves = iter(["a2a3", "a4a3", "quit"])  # second move is illegal (blocked) and re-asked
    play(input_fn=lambda _: next(moves), output_fn=lambda _: None,
         on_position=lambda state, last: seen.append((state.ply, last)))
    assert seen == [(0, None), (1, encode_move(parse_square("a2"), parse_square("a3")))]
