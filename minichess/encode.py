"""Network-facing encoding: observations, canonical actions, legal masks, mirror symmetry.

Everything is expressed from the point of view of the side to move: for Black the board
is flipped vertically (rank r -> 4 - r, files unchanged) and "own"/"opponent" pieces are
swapped. The start position is file-symmetric (kings on e, queens on d), so Black to move
on the initial position encodes exactly like White to move.
"""
from __future__ import annotations

import numpy as np

from .game import (
    EMPTY, NUM_ACTIONS, NUM_SQ, SIZE, WHITE, State, color_of, kind_of, legal_moves,
)

NUM_PLANES = 12  # own P N B R Q K, then opponent P N B R Q K


def flip_square(sq: int) -> int:
    """Vertical flip (perspective change for Black)."""
    r, f = divmod(sq, SIZE)
    return (SIZE - 1 - r) * SIZE + f


def mirror_square(sq: int) -> int:
    """Left-right mirror (data augmentation; the rules are file-symmetric)."""
    r, f = divmod(sq, SIZE)
    return r * SIZE + (SIZE - 1 - f)


FLIP_SQ = np.array([flip_square(s) for s in range(NUM_SQ)])
MIRROR_SQ = np.array([mirror_square(s) for s in range(NUM_SQ)])
FLIP_ACTION = np.array([flip_square(a // NUM_SQ) * NUM_SQ + flip_square(a % NUM_SQ)
                        for a in range(NUM_ACTIONS)])
MIRROR_ACTION = np.array([mirror_square(a // NUM_SQ) * NUM_SQ + mirror_square(a % NUM_SQ)
                          for a in range(NUM_ACTIONS)])


def to_canonical_action(action: int, turn: int) -> int:
    """Real action -> index in the mover's perspective. It is its own inverse."""
    return action if turn == WHITE else int(FLIP_ACTION[action])


from_canonical_action = to_canonical_action


def encode_state(state: State) -> np.ndarray:
    """uint8 array (12, 5, 5) indexed [plane, rank, file], from the side to move's view."""
    obs = np.zeros((NUM_PLANES, NUM_SQ), dtype=np.uint8)
    turn = state.turn
    for sq, piece in enumerate(state.squares):
        if piece == EMPTY:
            continue
        plane = kind_of(piece) - 1 + (0 if color_of(piece) == turn else 6)
        obs[plane, sq if turn == WHITE else FLIP_SQ[sq]] = 1
    return obs.reshape(NUM_PLANES, SIZE, SIZE)


def legal_mask(state: State) -> np.ndarray:
    """bool (625,) of legal moves, in canonical (mover's perspective) indices."""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for action in legal_moves(state):
        mask[to_canonical_action(action, state.turn)] = True
    return mask


def mirror_obs(obs: np.ndarray) -> np.ndarray:
    """Mirror along the file axis; works for (12,5,5) and batched (B,12,5,5)."""
    return np.ascontiguousarray(obs[..., ::-1])


def mirror_policy(pi: np.ndarray) -> np.ndarray:
    """Mirror a policy / mask over canonical actions; works for (625,) and (B,625)."""
    return pi[..., MIRROR_ACTION]
