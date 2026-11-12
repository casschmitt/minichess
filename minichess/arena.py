"""Players and matches, used to monitor training and to play against the trained model."""
from __future__ import annotations

from concurrent.futures import Executor
from typing import Callable, Optional

import numpy as np
import torch

from .game import MAX_PLY, WHITE, State, apply, initial_state, legal_moves, status
from .mcts import MCTS, net_evaluator
from .model import PolicyValueNet

Player = Callable[[State], int]


def random_player(rng: Optional[np.random.Generator] = None) -> Player:
    rng = rng or np.random.default_rng()
    return lambda state: int(rng.choice(legal_moves(state)))


def mcts_player(evaluator, sims: int = 30, rng: Optional[np.random.Generator] = None,
                temp_plies: int = 4, temperature: float = 1.0, max_ply: int = MAX_PLY) -> Player:
    """MCTS player: samples (``temperature``) for the first ``temp_plies`` plies, then plays
    the most-visited move. Use ``temp_plies=0`` for fully deterministic strength."""
    mcts = MCTS(evaluator, sims=sims, max_ply=max_ply, rng=rng)

    def choose(state: State) -> int:
        root = mcts.search(state, add_noise=False)
        return mcts.select_action(root, temperature if state.ply < temp_plies else 0.0)

    return choose


def net_player(net: PolicyValueNet, **kwargs) -> Player:
    return mcts_player(net_evaluator(net), **kwargs)


def play_one(white: Player, black: Player, max_ply: int = MAX_PLY) -> int:
    """Play a game; returns +1 (White wins), -1 (Black wins) or 0 (draw)."""
    state = initial_state()
    while True:
        result, _ = status(state, max_ply)
        if result is not None:
            return result
        action = (white if state.turn == WHITE else black)(state)
        assert action in legal_moves(state), "player returned an illegal move"
        state = apply(state, action)


def _rebuild(state_dict, channels, blocks) -> PolicyValueNet:
    net = PolicyValueNet(channels, blocks)
    net.load_state_dict(state_dict)
    net.eval()
    return net


def _match_worker(payload) -> dict:
    torch.set_num_threads(1)
    sd_a, ch_a, bl_a, sd_b, ch_b, bl_b, games, sims, seed, max_ply, temp_plies = payload
    rng = np.random.default_rng(seed)
    a = net_player(_rebuild(sd_a, ch_a, bl_a), sims=sims, rng=rng, temp_plies=temp_plies,
                   max_ply=max_ply)
    b = (random_player(rng) if sd_b is None else
         net_player(_rebuild(sd_b, ch_b, bl_b), sims=sims, rng=rng, temp_plies=temp_plies,
                    max_ply=max_ply))
    return play_match(a, b, games, max_ply)


def parallel_match(net_a: PolicyValueNet, net_b: Optional[PolicyValueNet], games: int, sims: int,
                   seed_seq: np.random.SeedSequence, executor: Optional[Executor] = None,
                   workers: int = 1, max_ply: int = MAX_PLY, temp_plies: int = 4) -> dict:
    """Match net_a against net_b (or a random mover if None); games are split over workers.
    Score is from net_a's side."""
    workers = max(1, workers) if executor is not None else 1
    counts = [c for c in (games // workers + (i < games % workers) for i in range(workers)) if c]
    seeds = seed_seq.spawn(len(counts))

    def sd(net):
        return None if net is None else {k: v.detach().cpu() for k, v in net.state_dict().items()}

    payloads = [(sd(net_a), net_a.channels, net_a.blocks, sd(net_b),
                 getattr(net_b, "channels", 0), getattr(net_b, "blocks", 0),
                 c, sims, s, max_ply, temp_plies) for c, s in zip(counts, seeds)]
    if executor is None:
        parts = [_match_worker(p) for p in payloads]
    else:
        parts = [f.result() for f in [executor.submit(_match_worker, p) for p in payloads]]
    total = {k: sum(p[k] for p in parts) for k in ("wins", "losses", "draws")}
    total["score"] = (total["wins"] + 0.5 * total["draws"]) / max(1, games)
    return total


def play_match(a: Player, b: Player, games: int, max_ply: int = MAX_PLY) -> dict:
    """``a`` plays White in even-numbered games and Black in odd ones. Score is from a's side."""
    wins = losses = draws = 0
    for g in range(games):
        result = play_one(a, b, max_ply) if g % 2 == 0 else -play_one(b, a, max_ply)
        wins += result == 1
        losses += result == -1
        draws += result == 0
    return {"wins": wins, "losses": losses, "draws": draws,
            "score": (wins + 0.5 * draws) / max(1, games)}
