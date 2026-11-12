import numpy as np
import torch
import torch.nn.functional as F

from minichess.encode import encode_state, legal_mask
from minichess.game import initial_state
from minichess.model import (
    PolicyValueNet, load_checkpoint, load_model, predict, predict_batch, save_checkpoint,
)


def test_forward_shapes_and_value_range():
    net = PolicyValueNet()
    x = torch.from_numpy(np.stack([encode_state(initial_state())] * 4))
    logits, value = net(x)
    assert logits.shape == (4, 625) and value.shape == (4,)
    assert (value.abs() <= 1).all()


def test_parameter_count_is_small():
    n = sum(p.numel() for p in PolicyValueNet().parameters())
    assert 30_000 < n < 200_000


def test_predict_is_masked_softmax_over_legal_moves():
    s = initial_state()
    priors, value = predict(PolicyValueNet(), encode_state(s), legal_mask(s))
    mask = legal_mask(s)
    assert priors.shape == (625,) and abs(priors.sum() - 1) < 1e-5
    assert (priors[~mask] == 0).all() and (priors[mask] > 0).all()
    assert -1 <= value <= 1


def test_predict_batch_matches_single():
    net = PolicyValueNet()
    s = initial_state()
    obs, mask = encode_state(s), legal_mask(s)
    p1, v1 = predict(net, obs, mask)
    pb, vb = predict_batch(net, np.stack([obs, obs]), np.stack([mask, mask]))
    assert np.allclose(pb[1], p1, atol=1e-6) and abs(vb[1] - v1) < 1e-6


def test_checkpoint_roundtrip(tmp_path):
    net = PolicyValueNet(channels=16, blocks=2)
    s = initial_state()
    before = predict(net, encode_state(s), legal_mask(s))
    path = tmp_path / "net.pt"
    opt = torch.optim.Adam(net.parameters())
    save_checkpoint(path, net, opt, iteration=7, note="x")
    ckpt = load_checkpoint(path)
    assert ckpt["iteration"] == 7 and ckpt["extra"] == {"note": "x"}
    loaded = load_model(path)
    assert (loaded.channels, loaded.blocks) == (16, 2)
    after = predict(loaded, encode_state(s), legal_mask(s))
    assert np.allclose(before[0], after[0], atol=1e-6) and abs(before[1] - after[1]) < 1e-6


def test_can_overfit_a_fixed_batch():
    torch.manual_seed(0)
    net = PolicyValueNet(channels=16, blocks=1)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    s = initial_state()
    x = torch.from_numpy(np.stack([encode_state(s)] * 8))
    pi = torch.zeros(8, 625)
    pi[:, np.flatnonzero(legal_mask(s))[0]] = 1.0     # target: one particular move
    z = torch.full((8,), 0.5)

    def step():
        net.train()
        logits, v = net(x)
        loss = -(pi * F.log_softmax(logits, -1)).sum(-1).mean() + F.mse_loss(v, z)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.item()

    first = step()
    for _ in range(50):
        last = step()
    assert last < first * 0.2
