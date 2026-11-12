"""GUI-independent game session: current position, legal targets, moves, undo."""
from __future__ import annotations

from typing import Optional

from .game import (
    MAX_PLY, State, apply, decode_move, encode_move, initial_state, legal_moves, status,
)


class GameController:
    def __init__(self, state: Optional[State] = None, max_ply: int = MAX_PLY):
        self._start = state or initial_state()
        self.max_ply = max_ply
        self.reset()

    # -- session state -------------------------------------------------------
    def reset(self) -> None:
        self.state = self._start
        self.last_action: Optional[int] = None
        self._history: list[tuple[State, Optional[int]]] = []
        self._refresh()

    def _refresh(self) -> None:
        self._legal = legal_moves(self.state)
        self.result, self.reason = status(self.state, self.max_ply)

    @property
    def game_over(self) -> bool:
        return self.result is not None

    # -- queries ---------------------------------------------------------------
    def targets_from(self, sq: int) -> list[int]:
        """Squares the piece on ``sq`` may legally move to (empty if none / game over)."""
        if self.game_over:
            return []
        return [to for frm, to in map(decode_move, self._legal) if frm == sq]

    # -- actions ---------------------------------------------------------------
    def try_move(self, frm: int, to: int) -> bool:
        """Play frm->to if legal. Returns True if the move was made."""
        action = encode_move(frm, to)
        if self.game_over or action not in self._legal:
            return False
        self._history.append((self.state, self.last_action))
        self.state = apply(self.state, action)
        self.last_action = action
        self._refresh()
        return True

    def undo(self) -> bool:
        if not self._history:
            return False
        self.state, self.last_action = self._history.pop()
        self._refresh()
        return True
