import random

import numpy as np

from minichess.encode import (
    FLIP_ACTION, FLIP_SQ, MIRROR_ACTION, MIRROR_SQ, encode_state, from_canonical_action,
    legal_mask, mirror_obs, mirror_policy, to_canonical_action,
)
from minichess.game import (
    BLACK, WHITE, State, apply, encode_move, initial_state, legal_moves, parse_square as sq,
    status,
)


def mirrored_state(s: State) -> State:
    """Same position reflected left-right (file a <-> e)."""
    return State(tuple(s.squares[MIRROR_SQ[i]] for i in range(25)), s.turn, s.ply)


def test_tables_are_involutions():
    for table in (FLIP_SQ, MIRROR_SQ, FLIP_ACTION, MIRROR_ACTION):
        assert (table[table] == np.arange(len(table))).all()


def test_action_conversion_roundtrip():
    a = encode_move(sq("b1"), sq("c3"))
    assert to_canonical_action(a, WHITE) == a
    c = to_canonical_action(a, BLACK)
    assert c != a and from_canonical_action(c, BLACK) == a


def test_observation_shape_and_piece_count():
    obs = encode_state(initial_state())
    assert obs.shape == (12, 5, 5) and obs.dtype == np.uint8
    assert obs.sum() == 20 and obs[:6].sum() == 10 and obs[6:].sum() == 10


def test_plane_assignment_for_white():
    obs = encode_state(initial_state())
    assert obs[0, 1, 0] == 1       # own pawn on a2 (rank index 1, file 0)
    assert obs[5, 0, 4] == 1       # own king on e1
    assert obs[6, 3, 0] == 1       # opponent pawn on a4
    assert obs[11, 4, 4] == 1      # opponent king on e5


def test_black_to_move_on_initial_position_encodes_like_white():
    start = initial_state()
    black = State(start.squares, BLACK)
    assert (encode_state(black) == encode_state(start)).all()
    assert (legal_mask(black) == legal_mask(start)).all()   # canonical moves match too


def test_legal_mask_initial_has_seven_moves():
    mask = legal_mask(initial_state())
    assert mask.dtype == bool and mask.shape == (625,) and mask.sum() == 7


def test_mask_in_canonical_indices_matches_real_actions_for_black():
    s = apply(initial_state(), encode_move(sq("a2"), sq("a3")))   # Black to move
    mask = legal_mask(s)
    real = {to_canonical_action(a, BLACK) for a in legal_moves(s)}
    assert set(np.flatnonzero(mask)) == real


def test_mirror_symmetry_along_random_games():
    rng = random.Random(1)
    for _ in range(20):
        s = initial_state()
        for _ in range(60):
            if status(s)[0] is not None:
                break
            m = mirrored_state(s)
            assert (encode_state(m) == mirror_obs(encode_state(s))).all()
            assert (legal_mask(m) == mirror_policy(legal_mask(s))).all()
            s = apply(s, rng.choice(legal_moves(s)))


def test_mirror_twice_is_identity():
    obs = encode_state(initial_state())
    assert (mirror_obs(mirror_obs(obs)) == obs).all()
    pi = np.random.default_rng(0).random(625)
    assert (mirror_policy(mirror_policy(pi)) == pi).all()


def test_mirror_works_on_batches():
    obs = np.stack([encode_state(initial_state())] * 3)
    pi = np.random.default_rng(0).random((3, 625))
    assert mirror_obs(obs).shape == (3, 12, 5, 5)
    assert (mirror_policy(pi)[1] == mirror_policy(pi[1])).all()
