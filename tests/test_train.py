import json

import numpy as np
import torch

from minichess.arena import parallel_match
from minichess.encode import encode_state, legal_mask
from minichess.game import initial_state
from minichess.model import PolicyValueNet, load_checkpoint
from minichess.replay import ReplayBuffer
from minichess.train import latest_checkpoint, main, train_steps

TINY = ["--channels", "8", "--blocks", "1", "--games", "4", "--sims", "4", "--steps", "5",
        "--batch", "8", "--eval-games", "2", "--eval-sims", "4", "--max-ply", "30",
        "--compare-every", "1", "--compare-games", "2", "--no-final-eval", "--buffer", "500"]


def read_log(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_train_steps_reduce_loss_on_a_fixed_buffer():
    torch.manual_seed(0)
    s = initial_state()
    buf = ReplayBuffer(16, rng=np.random.default_rng(0))
    pi = legal_mask(s).astype(np.float32)
    buf.add(np.stack([encode_state(s)] * 16), np.stack([pi / pi.sum()] * 16),
            np.full(16, 0.5, np.float32))
    net = PolicyValueNet(channels=16, blocks=1)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    first = sum(train_steps(net, opt, buf, 1, 16))
    later = sum(train_steps(net, opt, buf, 60, 16))
    assert later < first * 0.5


def test_training_run_writes_checkpoints_and_log(tmp_path):
    d = tmp_path / "ck"
    main(TINY + ["--iterations", "2", "--workers", "2", "--checkpoint-dir", str(d),
                 "--time-budget", "5"])
    for name in ("iter_000.pt", "iter_001.pt", "iter_002.pt", "best.pt", "buffer.npz", "log.jsonl"):
        assert (d / name).exists(), name
    rows = read_log(d / "log.jsonl")
    assert [r["iter"] for r in rows] == [1, 2]
    for r in rows:
        assert np.isfinite(r["policy_loss"]) and np.isfinite(r["value_loss"])
        assert r["games"] == 4 and 0 <= r["vs_random"] <= 1 and r["vs_best"] is not None
    assert load_checkpoint(d / "iter_002.pt")["iteration"] == 2


def test_resume_continues_from_the_latest_iteration(tmp_path):
    d = tmp_path / "ck"
    main(TINY + ["--iterations", "1", "--workers", "1", "--checkpoint-dir", str(d)])
    assert latest_checkpoint(d)[1] == 1
    main(TINY + ["--iterations", "2", "--workers", "1", "--checkpoint-dir", str(d), "--resume"])
    assert latest_checkpoint(d)[1] == 2
    assert [r["iter"] for r in read_log(d / "log.jsonl")] == [1, 2]


def test_time_budget_stops_after_the_first_iteration(tmp_path):
    d = tmp_path / "ck"
    main(TINY + ["--iterations", "50", "--workers", "1", "--checkpoint-dir", str(d),
                 "--time-budget", "0"])
    assert [r["iter"] for r in read_log(d / "log.jsonl")] == [1]


def test_parallel_match_inline_vs_net_and_random():
    a, b = PolicyValueNet(8, 1), PolicyValueNet(8, 1)
    r = parallel_match(a, b, 2, 4, np.random.SeedSequence(0), max_ply=30)
    assert r["wins"] + r["losses"] + r["draws"] == 2
    r = parallel_match(a, None, 2, 4, np.random.SeedSequence(0), max_ply=30)
    assert r["wins"] + r["losses"] + r["draws"] == 2
