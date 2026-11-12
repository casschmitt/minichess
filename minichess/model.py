"""Policy + value network for 5x5 mini chess (PyTorch)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encode import NUM_PLANES
from .game import NUM_ACTIONS, SIZE


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class PolicyValueNet(nn.Module):
    """Input (B,12,5,5) in the mover's perspective -> (policy logits (B,625), value (B,))."""

    def __init__(self, channels: int = 32, blocks: int = 3):
        super().__init__()
        self.channels, self.blocks = channels, blocks
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(),
        )
        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU(),
            nn.Flatten(), nn.Linear(2 * SIZE * SIZE, NUM_ACTIONS),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False), nn.BatchNorm2d(1), nn.ReLU(),
            nn.Flatten(), nn.Linear(SIZE * SIZE, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh(),
        )

    def forward(self, x):
        h = self.tower(self.stem(x.float()))
        return self.policy_head(h), self.value_head(h).squeeze(-1)


@torch.no_grad()
def predict_batch(net: PolicyValueNet, obs: np.ndarray, mask: np.ndarray):
    """obs (B,12,5,5), mask bool (B,625) -> (priors (B,625) float32, values (B,) float32).

    Priors are a softmax over the legal moves only (zero elsewhere). Puts ``net`` in eval mode.
    """
    if net.training:
        net.eval()
    logits, values = net(torch.from_numpy(obs))
    logits = logits.masked_fill(~torch.from_numpy(mask), float("-inf"))
    return F.softmax(logits, dim=-1).numpy(), values.numpy()


def predict(net: PolicyValueNet, obs: np.ndarray, mask: np.ndarray):
    """Single position: obs (12,5,5), mask (625,) -> (priors (625,), value float)."""
    priors, values = predict_batch(net, obs[None], mask[None])
    return priors[0], float(values[0])


def save_checkpoint(path, net: PolicyValueNet, optimizer=None, iteration: int = 0, **extra) -> None:
    torch.save({
        "channels": net.channels, "blocks": net.blocks, "model": net.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "iteration": iteration, "extra": extra,
    }, path)


def load_checkpoint(path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_model(path) -> PolicyValueNet:
    ckpt = load_checkpoint(path)
    net = PolicyValueNet(ckpt["channels"], ckpt["blocks"])
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net
