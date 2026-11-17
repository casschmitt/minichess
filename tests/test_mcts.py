import numpy as np
import pytest

from minichess.encode import legal_mask
from minichess.game import (
    BLACK, WHITE, State, apply, initial_state, legal_moves, move_name, status,
)
from minichess.mcts import MCTS, net_evaluator, uniform_evaluator
from minichess.model import PolicyValueNet


def board(diagram: str, turn=WHITE) -> State:
    rows = diagram.split()
    return State.from_string("".join(reversed(rows)) + "wb"[turn])


def make(sims=100, **kw) -> MCTS:
    return MCTS(uniform_evaluator, sims=sims, rng=np.random.default_rng(0), **kw)


def test_finds_mate_in_one_with_uniform_evaluator():
    s = board("""
        k....
        ....Q
        K....
        .....
        .....
    """)
    # sanity: this position really contains mating moves (Qb4#, Qa4#, Qe5#, ...)
    assert any(status(apply(s, a))[1] == "checkmate" for a in legal_moves(s))
    m = make(sims=100)
    action = m.select_action(m.search(s))
    assert status(apply(s, action)) == (1, "checkmate"), move_name(action)


def test_black_finds_mate_in_one_too():
    # vertical mirror of the White position with colours swapped, Black to move (Qb2#, ...)
    s = board("""
        .....
        .....
        k....
        ....q
        K....
    """, turn=BLACK)
    assert any(status(apply(s, a))[1] == "checkmate" for a in legal_moves(s))
    m = make(sims=100)
    action = m.select_action(m.search(s))
    assert status(apply(s, action)) == (-1, "checkmate"), move_name(action)


def test_visit_policy_is_a_distribution_on_legal_moves():
    for turn in (WHITE, BLACK):
        s = State(initial_state().squares, turn)
        m = make(sims=40)
        root = m.search(s)
        pi = m.visit_policy(root)
        assert pi.shape == (625,) and abs(pi.sum() - 1) < 1e-5
        assert (pi[~legal_mask(s)] == 0).all()
        assert root.N.sum() == 40


def test_select_action_is_always_legal_for_any_temperature():
    s = initial_state()
    m = MCTS(net_evaluator(PolicyValueNet()), sims=10, rng=np.random.default_rng(3))
    root = m.search(s, add_noise=True)
    for temperature in (0.0, 0.5, 1.0):
        for _ in range(10):
            assert m.select_action(root, temperature) in legal_moves(s)


def test_noise_changes_root_priors_but_keeps_a_distribution():
    s = initial_state()
    m = make(sims=1)
    plain = m.search(s, add_noise=False).P.copy()
    noisy = m.search(s, add_noise=True).P
    assert not np.allclose(plain, noisy) and abs(noisy.sum() - 1) < 1e-9


def test_search_is_deterministic_without_noise():
    s = initial_state()
    a = MCTS(uniform_evaluator, sims=50, rng=np.random.default_rng(1)).search(s).N
    b = MCTS(uniform_evaluator, sims=50, rng=np.random.default_rng(2)).search(s).N
    assert (a == b).all()


def test_terminal_root_raises():
    mate = board("""
        k....
        .Q...
        ..K..
        .....
        .....
    """, turn=BLACK)
    with pytest.raises(ValueError):
        make().search(mate)


def test_ply_cap_is_treated_as_a_draw_inside_the_tree():
    s = State(initial_state().squares, WHITE, ply=99)
    m = make(sims=20, max_ply=100)   # every reply reaches ply 100 -> draw, value 0
    root = m.search(s)
    assert np.allclose(root.W, 0)
