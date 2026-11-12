"""Self-play data generation: MCTS games -> (observation, visit policy, outcome) examples."""
from __future__ import annotations

from concurrent.futures import Executor
from typing import Optional

import numpy as np
import torch

from .encode import encode_state
from .game import WHITE, State, apply, initial_state
from .mcts import MCTS, net_evaluator
from .model import PolicyValueNet

SELFPLAY_MAX_PLY = 100


def result_for_mover(result: int, turn: int) -> int:
    """Convert a game result (+1 White wins, -1 Black wins, 0 draw) to the mover's perspective."""
    return result if turn == WHITE else -result


def play_game(evaluator, sims: int, rng: np.random.Generator, max_ply: int = SELFPLAY_MAX_PLY,
              temp_plies: int = 6, c_puct: float = 1.5, add_noise: bool = True,
              start: Optional[State] = None):
    """Play one self-play game. Returns (samples, result); samples = [(obs, pi, z), ...].

    The first ``temp_plies`` moves are sampled proportionally to visit counts (exploration),
    later ones are the most-visited move. ``z`` is the final outcome seen by whoever moved.
    """
    state = start or initial_state()
    mcts = MCTS(evaluator, sims=sims, c_puct=c_puct, max_ply=max_ply, rng=rng)
    history = []
    while True:
        result = mcts.result(state)
        if result is not None:
            break
        root = mcts.search(state, add_noise=add_noise)
        history.append((encode_state(state), mcts.visit_policy(root), state.turn))
        temperature = 1.0 if state.ply < temp_plies else 0.0
        state = apply(state, mcts.select_action(root, temperature))
    samples = [(obs, pi, float(result_for_mover(result, turn))) for obs, pi, turn in history]
    return samples, result


def run_games(net: PolicyValueNet, games: int, sims: int, seed, max_ply: int = SELFPLAY_MAX_PLY,
              temp_plies: int = 6):
    """Play ``games`` self-play games in this process. Returns (obs, pi, z, stats)."""
    rng = np.random.default_rng(seed)
    evaluator = net_evaluator(net)
    obs, pi, z = [], [], []
    stats = {"games": 0, "white": 0, "black": 0, "draws": 0, "plies": 0}
    for _ in range(games):
        samples, result = play_game(evaluator, sims, rng, max_ply, temp_plies)
        obs += [s[0] for s in samples]
        pi += [s[1] for s in samples]
        z += [s[2] for s in samples]
        stats["games"] += 1
        stats["plies"] += len(samples)
        stats["white" if result == 1 else "black" if result == -1 else "draws"] += 1
    return np.stack(obs), np.stack(pi), np.array(z, dtype=np.float32), stats


def _worker(payload):
    torch.set_num_threads(1)
    state_dict, channels, blocks, games, sims, seed, max_ply, temp_plies = payload
    net = PolicyValueNet(channels, blocks)
    net.load_state_dict(state_dict)
    net.eval()
    return run_games(net, games, sims, seed, max_ply, temp_plies)


def generate(net: PolicyValueNet, games: int, sims: int, seed_seq: np.random.SeedSequence,
             executor: Optional[Executor] = None, workers: int = 1,
             max_ply: int = SELFPLAY_MAX_PLY, temp_plies: int = 6):
    """Generate ``games`` games, split over ``workers`` processes if an executor is given."""
    workers = max(1, workers) if executor is not None else 1
    counts = [games // workers + (i < games % workers) for i in range(workers)]
    counts = [c for c in counts if c > 0]
    seeds = seed_seq.spawn(len(counts))
    if executor is None:
        parts = [run_games(net, c, sims, s, max_ply, temp_plies) for c, s in zip(counts, seeds)]
    else:
        state_dict = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        futures = [executor.submit(_worker, (state_dict, net.channels, net.blocks, c, sims, s,
                                             max_ply, temp_plies))
                   for c, s in zip(counts, seeds)]
        parts = [f.result() for f in futures]
    obs = np.concatenate([p[0] for p in parts])
    pi = np.concatenate([p[1] for p in parts])
    z = np.concatenate([p[2] for p in parts])
    stats = {k: sum(p[3][k] for p in parts) for k in parts[0][3]}
    return obs, pi, z, stats
