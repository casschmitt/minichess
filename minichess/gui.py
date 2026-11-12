"""Drag-and-drop mini chess window (matplotlib).  Run:  python -m minichess.gui

Drag a piece onto a highlighted square to move it (illegal drops snap back), or click a
piece and then click its destination. Keys: u = undo, r = restart, q = quit.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

from .controller import GameController
from .game import BLACK, SIZE, WHITE, State, decode_move
from .render_png import paint_position, piece_artist

HINT = "drag a piece to move  ·  u undo  ·  r restart  ·  q quit"


class BoardGUI:
    """Interactive board on a matplotlib Figure (any backend). Keep a reference to it:
    matplotlib only holds weak references to the event callbacks."""

    def __init__(self, fig, controller: Optional[GameController] = None,
                 ai_player: Optional[Callable[[State], int]] = None, human_color: int = WHITE):
        self.fig = fig
        self.canvas = fig.canvas
        self.ctrl = controller or GameController()
        self.ai_player = ai_player          # state -> action; None = two humans
        self.human_color = human_color
        self._timer = None
        self.ax = fig.add_subplot()
        fig.subplots_adjust(left=0.08, right=0.92, bottom=0.07, top=0.88)
        fig.text(0.5, 0.015, HINT, ha="center", fontsize=10, color="#666")

        self.selected: Optional[int] = None   # square of the picked-up / selected piece
        self._drag = None                     # (origin square, floating Text artist)
        self._bg = None                       # cached background for blitting
        for event, handler in (
            ("button_press_event", self._on_press),
            ("motion_notify_event", self._on_motion),
            ("button_release_event", self._on_release),
            ("key_press_event", self._on_key),
        ):
            self.canvas.mpl_connect(event, handler)
        self.redraw()
        self._schedule_ai()                 # the AI opens if the human plays Black

    # -- AI opponent -----------------------------------------------------------
    def _ai_to_move(self) -> bool:
        return (self.ai_player is not None and not self.ctrl.game_over
                and self.ctrl.state.turn != self.human_color)

    def _schedule_ai(self) -> None:
        """Let the AI reply shortly after the human's move is drawn (keeps the UI responsive)."""
        if not self._ai_to_move():
            return
        self._timer = self.canvas.new_timer(interval=100)
        self._timer.single_shot = True
        self._timer.add_callback(self._ai_move)
        self._timer.start()

    def _ai_move(self) -> None:
        if not self._ai_to_move():
            return
        self.ctrl.try_move(*decode_move(self.ai_player(self.ctrl.state)))
        self.selected = None
        self.redraw()

    # -- helpers ---------------------------------------------------------------
    def _data(self, event) -> tuple[float, float]:
        """Cursor position in board coordinates (works even outside the axes)."""
        x, y = self.ax.transData.inverted().transform((event.x, event.y))
        return float(x), float(y)

    def _square_at(self, event) -> Optional[int]:
        x, y = self._data(event)
        f, r = math.floor(x), math.floor(y)
        return r * SIZE + f if 0 <= f < SIZE and 0 <= r < SIZE else None

    def redraw(self, hidden: Optional[int] = None) -> None:
        targets = self.ctrl.targets_from(self.selected) if self.selected is not None else ()
        paint_position(self.ax, self.ctrl.state, self.ctrl.last_action,
                       hidden_sq=hidden, selected_sq=self.selected, targets=targets)
        self.canvas.draw_idle()

    def _cancel_drag(self) -> None:
        self._drag = None
        self._bg = None

    # -- drag and drop ---------------------------------------------------------
    def _begin_drag(self, event, sq: int) -> None:
        self.selected = sq
        self.redraw(hidden=sq)
        x, y = self._data(event)
        blit = getattr(self.canvas, "supports_blit", False)
        art = piece_artist(self.ax, self.ctrl.state.squares[sq], x, y, animated=blit)
        self._drag = (sq, art)
        if blit:
            self.canvas.draw()  # render the static scene (animated artists are skipped)
            self._bg = self.canvas.copy_from_bbox(self.fig.bbox)
            self._blit_floating()
        else:
            self.canvas.draw_idle()

    def _blit_floating(self) -> None:
        self.canvas.restore_region(self._bg)
        self.ax.draw_artist(self._drag[1])
        self.canvas.blit(self.fig.bbox)

    def _on_press(self, event) -> None:
        if event.button != 1 or self._ai_to_move():
            return
        sq = self._square_at(event)
        if sq is None:
            self.selected = None
            self.redraw()
        elif self.selected is not None and sq in self.ctrl.targets_from(self.selected):
            self.ctrl.try_move(self.selected, sq)   # click-click move
            self.selected = None
            self.redraw()
            self._schedule_ai()
        elif self.ctrl.targets_from(sq):            # own piece that can move: pick it up
            self._begin_drag(event, sq)
        else:
            self.selected = None
            self.redraw()

    def _on_motion(self, event) -> None:
        if self._drag is None:
            return
        self._drag[1].set_position(self._data(event))
        if self._bg is not None:
            self._blit_floating()
        else:
            self.canvas.draw_idle()

    def _on_release(self, event) -> None:
        if self._drag is None or event.button != 1:
            return
        origin = self._drag[0]
        self._cancel_drag()
        sq = self._square_at(event)
        moved = sq is not None and sq != origin and self.ctrl.try_move(origin, sq)
        if moved or sq != origin:
            self.selected = None   # moved, or illegal drop / off-board: snap back and deselect
        # dropped back on its own square: stays selected, legal targets stay shown
        self.redraw()
        if moved:
            self._schedule_ai()

    # -- keyboard --------------------------------------------------------------
    def _on_key(self, event) -> None:
        if event.key == "u":
            self.ctrl.undo()
            if self.ai_player is not None and self.ctrl.state.turn != self.human_color:
                self.ctrl.undo()    # take back the AI's reply as well, so it is the human's turn
        elif event.key == "r":
            self.ctrl.reset()
        else:
            return
        self._cancel_drag()
        self.selected = None
        self.redraw()
        if event.key == "r":
            self._schedule_ai()


def main() -> None:
    import argparse

    import matplotlib

    parser = argparse.ArgumentParser(description="Drag-and-drop 5x5 mini chess.")
    parser.add_argument("--backend", default="TkAgg",
                        help="matplotlib backend: TkAgg (default window), QtAgg, or webagg "
                             "(opens in a browser; use it if the Tk window stays blank)")
    parser.add_argument("--ai", metavar="CHECKPOINT", help="play against this trained network")
    parser.add_argument("--human", choices=("white", "black"), default="white",
                        help="your colour when playing against the AI (default: white)")
    parser.add_argument("--sims", type=int, default=100,
                        help="MCTS simulations per AI move (more = stronger but slower)")
    args = parser.parse_args()

    ai_player = None
    if args.ai:
        import torch

        from .arena import net_player
        from .model import load_model

        torch.set_num_threads(1)   # single-position inference is fastest on one thread
        ai_player = net_player(load_model(args.ai), sims=args.sims, temp_plies=2)

    matplotlib.use(args.backend)  # must be chosen before importing pyplot
    matplotlib.rcParams["toolbar"] = "None"
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 6.6), dpi=110)
    fig.canvas.manager.set_window_title("Mini Chess")
    gui = BoardGUI(fig, ai_player=ai_player,  # noqa: F841  (keep alive: callbacks are weak refs)
                   human_color=WHITE if args.human == "white" else BLACK)

    if matplotlib.get_backend().lower() == "tkagg":
        # Some desktops (e.g. WSLg) don't paint a Tk window until it is redrawn after being
        # shown, so force a repaint once it is mapped and shortly after startup.
        widget = fig.canvas.get_tk_widget()
        widget.bind("<Map>", lambda e: fig.canvas.draw_idle(), add="+")
        widget.after(300, fig.canvas.draw_idle)
    plt.show()


if __name__ == "__main__":
    main()
