"""Fixed-size replay buffer of (observation, MCTS policy, outcome) training examples."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .encode import NUM_PLANES, mirror_obs, mirror_policy
from .game import NUM_ACTIONS, SIZE


class ReplayBuffer:
    def __init__(self, capacity: int = 30_000, rng: Optional[np.random.Generator] = None):
        self.capacity = capacity
        self.rng = rng or np.random.default_rng()
        self.obs = np.zeros((capacity, NUM_PLANES, SIZE, SIZE), dtype=np.uint8)
        self.pi = np.zeros((capacity, NUM_ACTIONS), dtype=np.float16)
        self.z = np.zeros(capacity, dtype=np.float16)
        self.size = 0
        self._ptr = 0

    def __len__(self) -> int:
        return self.size

    def add(self, obs: np.ndarray, pi: np.ndarray, z: np.ndarray) -> None:
        """Append a batch (N,12,5,5), (N,625), (N,); the oldest entries are overwritten."""
        n = len(obs)
        if n > self.capacity:                      # keep only the newest `capacity` rows
            obs, pi, z, n = obs[-self.capacity:], pi[-self.capacity:], z[-self.capacity:], self.capacity
        idx = (self._ptr + np.arange(n)) % self.capacity
        self.obs[idx], self.pi[idx], self.z[idx] = obs, pi, z
        self._ptr = (self._ptr + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(self, batch_size: int, augment: bool = True):
        """Random batch as (uint8 obs, float32 pi, float32 z); half are left-right mirrored."""
        idx = self.rng.integers(0, self.size, batch_size)
        obs, pi, z = self.obs[idx], self.pi[idx].astype(np.float32), self.z[idx].astype(np.float32)
        if augment:
            flip = self.rng.random(batch_size) < 0.5
            obs[flip] = mirror_obs(obs[flip])
            pi[flip] = mirror_policy(pi[flip])
        return obs, pi, z

    # -- persistence (for --resume) ---------------------------------------------
    def save(self, path) -> None:
        np.savez_compressed(path, obs=self.obs[:self.size], pi=self.pi[:self.size],
                            z=self.z[:self.size])

    def load(self, path) -> None:
        data = np.load(path)
        self.size = self._ptr = 0
        self.add(data["obs"], data["pi"], data["z"])
