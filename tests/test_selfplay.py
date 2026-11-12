import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from minichess.arena import mcts_player, play_match, play_one, random_player
from minichess.encode import encode_state, legal_mask, mirror_obs, mirror_policy
from minichess.game import BLACK, WHITE, State, initial_state, legal_moves
from minichess.mcts import uniform_evaluator
from minichess.model import PolicyValueNet
from minichess.replay import ReplayBuffer
from minichess.selfplay import generate, play_game, result_for_mover, run_games


def board(diagram: str, turn=WHITE) -> State:
    rows = diagram.split()
    return State.from_string("".join(reversed(rows)) + "wb"[turn])


# ---------------------------------------------------------------- outcome sign (z)

def test_result_for_mover():
    assert result_for_mover(1, WHITE) == 1 and result_for_mover(1, BLACK) == -1
    assert result_for_mover(-1, BLACK) == 1 and result_for_mover(-1, WHITE) == -1
    assert result_for_mover(0, WHITE) == 0 and result_for_mover(0, BLACK) == 0


def test_z_is_positive_for_the_side_that_mates_white():
    start = board("""
        k....
        ....Q
        K....
        .....
        .....
    """)
    samples, result = play_game(uniform_evaluator, sims=60, rng=np.random.default_rng(0),
                                temp_plies=0, start=start)
    assert result == 1 and len(samples) == 1
    assert samples[0][2] == 1.0


def test_z_is_positive_for_the_side_that_mates_black():
    start = board("""
        .....
        .....
        k....
        ....q
        K....
    """, turn=BLACK)
    samples, result = play_game(uniform_evaluator, sims=60, rng=np.random.default_rng(0),
                                temp_plies=0, start=start)
    assert result == -1 and len(samples) == 1
    assert samples[0][2] == 1.0          # Black moved and won -> +1 from Black's perspective


# ---------------------------------------------------------------- full self-play game

def test_play_game_produces_consistent_examples():
    samples, result = play_game(uniform_evaluator, sims=8, rng=np.random.default_rng(0),
                                max_ply=60)
    assert len(samples) >= 1
    assert (samples[0][0] == encode_state(initial_state())).all()
    for obs, pi, z in samples:
        assert obs.shape == (12, 5, 5) and pi.shape == (625,) and abs(pi.sum() - 1) < 1e-5
        assert z in (-1.0, 0.0, 1.0)
    zs = [s[2] for s in samples]
    if result == 0:
        assert all(z == 0 for z in zs)
    else:   # movers alternate, so the outcome flips sign from one example to the next
        assert all(zs[i] == -zs[i + 1] for i in range(len(zs) - 1))
        assert zs[-1] == 1.0              # the last mover is the winner (mate / pawn rush)


def test_first_example_policy_is_on_legal_moves():
    samples, _ = play_game(uniform_evaluator, sims=8, rng=np.random.default_rng(1), max_ply=20)
    assert (samples[0][1][~legal_mask(initial_state())] == 0).all()


def test_play_game_is_reproducible_with_seed():
    a, _ = play_game(uniform_evaluator, 8, np.random.default_rng(5), max_ply=30)
    b, _ = play_game(uniform_evaluator, 8, np.random.default_rng(5), max_ply=30)
    assert len(a) == len(b) and all((x[1] == y[1]).all() for x, y in zip(a, b))


def test_run_games_inline_and_generate_shapes():
    net = PolicyValueNet(channels=8, blocks=1)
    obs, pi, z, stats = generate(net, games=3, sims=4, seed_seq=np.random.SeedSequence(0),
                                 max_ply=30)
    assert obs.shape[1:] == (12, 5, 5) and pi.shape[1] == 625
    assert len(obs) == len(pi) == len(z) == stats["plies"]
    assert stats["games"] == 3 and stats["white"] + stats["black"] + stats["draws"] == 3


def test_generate_with_worker_processes():
    net = PolicyValueNet(channels=8, blocks=1)
    with ProcessPoolExecutor(2, mp_context=mp.get_context("spawn")) as pool:
        obs, pi, z, stats = generate(net, games=4, sims=4, seed_seq=np.random.SeedSequence(1),
                                     executor=pool, workers=2, max_ply=30)
    assert stats["games"] == 4 and len(obs) == len(pi) == len(z) == stats["plies"]


# ---------------------------------------------------------------- replay buffer

def test_replay_wraps_around_and_keeps_newest():
    buf = ReplayBuffer(capacity=10, rng=np.random.default_rng(0))
    obs = np.zeros((25, 12, 5, 5), dtype=np.uint8)
    pi = np.zeros((25, 625), dtype=np.float32)
    buf.add(obs, pi, np.arange(25, dtype=np.float32))
    assert len(buf) == 10
    assert sorted(buf.z.astype(int).tolist()) == list(range(15, 25))
    buf.add(obs[:3], pi[:3], np.array([100, 101, 102], dtype=np.float32))
    assert len(buf) == 10 and {100, 101, 102} <= set(buf.z.astype(int).tolist())


def test_replay_sample_augmentation_is_consistent():
    s = initial_state()
    obs0, pi0 = encode_state(s), legal_mask(s).astype(np.float32) / 7
    buf = ReplayBuffer(capacity=4, rng=np.random.default_rng(0))
    buf.add(obs0[None], pi0[None], np.array([1.0], dtype=np.float32))
    o, p, z = buf.sample(64)
    assert o.dtype == np.uint8 and p.dtype == np.float32 and (z == 1).all()
    seen_original = seen_mirrored = False
    for i in range(64):
        if (o[i] == obs0).all():
            assert np.allclose(p[i], pi0, atol=1e-3)
            seen_original = True
        else:
            assert (o[i] == mirror_obs(obs0)).all()
            assert np.allclose(p[i], mirror_policy(pi0), atol=1e-3)
            seen_mirrored = True
    assert seen_original and seen_mirrored


def test_replay_save_load_roundtrip(tmp_path):
    buf = ReplayBuffer(capacity=8, rng=np.random.default_rng(0))
    obs = np.random.default_rng(0).integers(0, 2, (5, 12, 5, 5)).astype(np.uint8)
    buf.add(obs, np.zeros((5, 625), np.float32), np.arange(5, dtype=np.float32))
    buf.save(tmp_path / "buf.npz")
    other = ReplayBuffer(capacity=8)
    other.load(tmp_path / "buf.npz")
    assert len(other) == 5 and (other.obs[:5] == obs).all()


# ---------------------------------------------------------------- arena

def test_random_player_plays_legal_moves():
    p = random_player(np.random.default_rng(0))
    s = initial_state()
    assert all(p(s) in legal_moves(s) for _ in range(20))


def test_play_match_totals_and_colour_alternation():
    rng = np.random.default_rng(0)
    res = play_match(random_player(rng), random_player(rng), games=6, max_ply=60)
    assert res["wins"] + res["losses"] + res["draws"] == 6
    assert 0 <= res["score"] <= 1


def test_play_one_with_mcts_player_terminates_legally():
    rng = np.random.default_rng(0)
    mc = mcts_player(uniform_evaluator, sims=6, rng=rng)
    assert play_one(mc, random_player(rng), max_ply=40) in (-1, 0, 1)
