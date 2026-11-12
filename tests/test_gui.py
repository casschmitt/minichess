"""Headless GUI tests: synthesize matplotlib mouse/key events on an Agg canvas."""
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from minichess.controller import GameController
from minichess.game import BLACK, WHITE, State, initial_state, legal_moves, parse_square as sq
from minichess.gui import BoardGUI


def make_gui(state=None) -> BoardGUI:
    fig = Figure(figsize=(6, 6.6), dpi=110)
    FigureCanvasAgg(fig)
    gui = BoardGUI(fig, GameController(state))
    fig.canvas.draw()
    return gui


def mouse(gui, name, square, button=1):
    f, r = square % 5, square // 5
    x, y = gui.ax.transData.transform((f + 0.5, r + 0.5))
    gui.canvas.callbacks.process(name, MouseEvent(name, gui.canvas, x, y, button=button))


def press(gui, s): mouse(gui, "button_press_event", sq(s))
def move(gui, s): mouse(gui, "motion_notify_event", sq(s), button=None)
def release(gui, s): mouse(gui, "button_release_event", sq(s))


def key(gui, k):
    gui.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", gui.canvas, k))


def drag(gui, a, b):
    press(gui, a)
    move(gui, b)
    release(gui, b)


def test_drag_legal_move():
    g = make_gui()
    press(g, "a2")
    assert g.selected == sq("a2") and g._drag is not None
    move(g, "a3")
    release(g, "a3")
    assert g.ctrl.state.turn == BLACK
    assert g.ctrl.state.squares[sq("a3")] == 1 and g.ctrl.state.squares[sq("a2")] == 0
    assert g.selected is None and g._drag is None


def test_drag_illegal_drop_snaps_back():
    g = make_gui()
    drag(g, "a2", "a4")           # pawns cannot double-step
    assert g.ctrl.state == initial_state()
    assert g.selected is None and g._drag is None


def test_drag_off_board_snaps_back():
    g = make_gui()
    press(g, "a2")
    x, y = g.ax.transData.transform((-3.0, -3.0))
    g.canvas.callbacks.process("button_release_event",
                               MouseEvent("button_release_event", g.canvas, x, y, button=1))
    assert g.ctrl.state == initial_state() and g.selected is None


def test_click_then_click_moves():
    g = make_gui()
    press(g, "b1"); release(g, "b1")
    assert g.selected == sq("b1") and g._drag is None   # picked up, waiting for a target
    press(g, "c3"); release(g, "c3")
    assert g.ctrl.state.squares[sq("c3")] == 2 and g.selected is None


def test_cannot_pick_up_opponent_or_blocked_piece():
    g = make_gui()
    press(g, "a4"); release(g, "a4")
    assert g.selected is None
    press(g, "a1"); release(g, "a1")
    assert g.selected is None and g.ctrl.state == initial_state()


def test_right_click_is_ignored():
    g = make_gui()
    mouse(g, "button_press_event", sq("a2"), button=3)
    assert g.selected is None and g._drag is None


def test_capture_by_drag_and_undo_restart_keys():
    g = make_gui()
    drag(g, "a2", "a3")
    drag(g, "b5", "a3")           # black knight takes the pawn
    assert g.ctrl.state.squares[sq("a3")] == 8 and g.ctrl.state.turn == WHITE
    key(g, "u")
    assert g.ctrl.state.squares[sq("a3")] == 1 and g.ctrl.state.turn == BLACK
    key(g, "r")
    assert g.ctrl.state == initial_state()


def test_input_locked_after_game_over():
    mate = State.from_string("".join(reversed(["k....", ".Q...", "..K..", ".....", "....."])) + "b")
    g = make_gui(mate)
    press(g, "a5"); release(g, "a5")
    assert g.selected is None and g.ctrl.game_over


# ---------------------------------------------------------------- AI opponent

def first_legal_ai(state):
    return legal_moves(state)[0]


def make_ai_gui(human_color=WHITE) -> BoardGUI:
    fig = Figure(figsize=(6, 6.6), dpi=110)
    FigureCanvasAgg(fig)
    gui = BoardGUI(fig, ai_player=first_legal_ai, human_color=human_color)
    fig.canvas.draw()
    return gui


def test_ai_replies_after_human_move():
    g = make_ai_gui(WHITE)
    drag(g, "a2", "a3")
    assert g.ctrl.state.turn == BLACK and g._ai_to_move()
    g._ai_move()                      # normally fired by a 100 ms timer
    assert g.ctrl.state.turn == WHITE and g.ctrl.state.ply == 2


def test_human_cannot_move_while_ai_is_to_move():
    g = make_ai_gui(WHITE)
    drag(g, "a2", "a3")               # now it is the AI's turn
    before = g.ctrl.state
    drag(g, "b5", "a3")               # trying to move the AI's knight
    assert g.ctrl.state == before and g.selected is None


def test_ai_opens_when_human_plays_black():
    g = make_ai_gui(BLACK)
    assert g._ai_to_move()
    g._ai_move()
    assert g.ctrl.state.turn == BLACK and g.ctrl.state.ply == 1


def test_undo_takes_back_both_human_and_ai_moves():
    g = make_ai_gui(WHITE)
    drag(g, "a2", "a3")
    g._ai_move()
    key(g, "u")
    assert g.ctrl.state == initial_state()


def test_ai_player_is_not_used_in_two_human_mode():
    g = make_gui()
    drag(g, "a2", "a3")
    assert not g._ai_to_move() and g.ctrl.state.turn == BLACK
