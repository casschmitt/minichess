"""PUCT Monte-Carlo tree search driven by a policy/value evaluator.

All values are from the point of view of the player to move at the node they belong to.
Edge statistics live in the parent node (arrays over its legal moves); when a value is
backed up one level the sign flips, because the parent's mover is the child's opponent.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .encode import FLIP_ACTION, NUM_ACTIONS, encode_state
from .game import MAX_PLY, WHITE, State, apply, legal_moves, status
from .model import PolicyValueNet, predict

# evaluator(obs (12,5,5), mask bool (625,)) -> (priors (625,) over legal moves, value float)
Evaluator = Callable[[np.ndarray, np.ndarray], tuple]


def net_evaluator(net: PolicyValueNet) -> Evaluator:
    return lambda obs, mask: predict(net, obs, mask)


def uniform_evaluator(obs: np.ndarray, mask: np.ndarray) -> tuple:
    """Uninformed evaluator (uniform priors, value 0): search relies on terminal states only."""
    return mask / mask.sum(), 0.0


class Node:
    __slots__ = ("state", "actions", "canon", "P", "N", "W", "children", "terminal_value")

    def __init__(self, state: State):
        self.state = state
        self.actions = self.canon = self.P = self.N = self.W = self.children = None
        self.terminal_value: Optional[float] = None


class MCTS:
    def __init__(self, evaluator: Evaluator, sims: int = 30, c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3, dirichlet_eps: float = 0.25,
                 max_ply: int = MAX_PLY, rng: Optional[np.random.Generator] = None):
        self.evaluator = evaluator
        self.sims = sims
        self.c_puct = c_puct
        self.alpha, self.eps = dirichlet_alpha, dirichlet_eps
        self.max_ply = max_ply
        self.rng = rng or np.random.default_rng()
        self._legal_cache: dict[bytes, list[int]] = {}

    # -- helpers ---------------------------------------------------------------
    def _legal(self, state: State) -> list[int]:
        key = state.pack()
        legal = self._legal_cache.get(key)
        if legal is None:
            if len(self._legal_cache) > 200_000:
                self._legal_cache.clear()
            legal = self._legal_cache[key] = legal_moves(state)
        return legal

    def _expand(self, node: Node) -> float:
        """Fill in a new node; returns its value for the player to move there."""
        state = node.state
        legal = self._legal(state)
        result, _ = status(state, self.max_ply, legal)
        if result is not None:
            node.terminal_value = float(result if state.turn == WHITE else -result)
            return node.terminal_value
        actions = np.array(legal)
        canon = actions if state.turn == WHITE else FLIP_ACTION[actions]
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        mask[canon] = True
        priors, value = self.evaluator(encode_state(state), mask)
        node.actions, node.canon = actions, canon
        node.P = np.asarray(priors, dtype=np.float64)[canon]
        node.N = np.zeros(len(actions))
        node.W = np.zeros(len(actions))
        node.children = [None] * len(actions)
        return float(value)

    def _select(self, node: Node) -> int:
        n_total = max(1.0, node.N.sum())
        q = np.divide(node.W, node.N, out=np.zeros_like(node.W), where=node.N > 0)
        u = self.c_puct * node.P * np.sqrt(n_total) / (1.0 + node.N)
        return int(np.argmax(q + u))

    def _simulate(self, root: Node) -> None:
        node, path = root, []
        while True:
            idx = self._select(node)
            path.append((node, idx))
            child = node.children[idx]
            if child is None:
                child = node.children[idx] = Node(apply(node.state, int(node.actions[idx])))
                value = self._expand(child)
                break
            if child.terminal_value is not None:
                value = child.terminal_value
                break
            node = child
        for node, idx in reversed(path):   # value is for the mover one level below `node`
            value = -value
            node.N[idx] += 1
            node.W[idx] += value

    # -- public API ------------------------------------------------------------
    def result(self, state: State) -> Optional[int]:
        """Game result at ``state`` (+1 White, -1 Black, 0 draw, None if ongoing), cached."""
        return status(state, self.max_ply, self._legal(state))[0]

    def search(self, state: State, add_noise: bool = False) -> Node:
        """Run ``sims`` simulations from ``state`` (must not be terminal); returns the root."""
        root = Node(state)
        self._expand(root)
        if root.terminal_value is not None:
            raise ValueError("cannot search a finished game")
        if add_noise and len(root.P) > 1:
            noise = self.rng.dirichlet([self.alpha] * len(root.P))
            root.P = (1 - self.eps) * root.P + self.eps * noise
        for _ in range(self.sims):
            self._simulate(root)
        return root

    @staticmethod
    def visit_policy(root: Node) -> np.ndarray:
        """Visit-count distribution over canonical actions (625,), the training target."""
        pi = np.zeros(NUM_ACTIONS, dtype=np.float32)
        pi[root.canon] = root.N / root.N.sum()
        return pi

    def select_action(self, root: Node, temperature: float = 0.0) -> int:
        """Real action. temperature 0 = most visited; otherwise sample proportional to N^(1/T)."""
        if temperature <= 1e-6:
            best = np.flatnonzero(root.N == root.N.max())
            return int(root.actions[self.rng.choice(best)])
        weights = root.N ** (1.0 / temperature)
        return int(root.actions[self.rng.choice(len(weights), p=weights / weights.sum())])
