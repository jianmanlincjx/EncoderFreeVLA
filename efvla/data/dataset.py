"""Frame-level dataset over the uint8 cache. One item = one timestep t of one episode."""
import json
import os
import shutil
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from .normalize import Normalizer


def ensure_shm_copy(cache_dir: str, shm_dir: str) -> str:
    """Copy the cache to /dev/shm once (atomic, safe if several jobs race). Returns the directory to read."""
    if not shm_dir:
        return cache_dir
    os.makedirs(shm_dir, exist_ok=True)
    for name in ("frames.npy", "meta.npz", "tasks.json", "stats.json"):
        src, dst = os.path.join(cache_dir, name), os.path.join(shm_dir, name)
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
            continue
        tmp = dst + f".tmp{os.getpid()}"
        t0 = time.time()
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        print(f"[data] copied {name} to {shm_dir} in {time.time() - t0:.0f}s", flush=True)
    return shm_dir


class LiberoFrames(Dataset):
    def __init__(self, data_dir: str, chunk: int):
        self.dir = data_dir
        self.chunk = chunk
        self.frames = np.load(os.path.join(data_dir, "frames.npy"), mmap_mode="r")
        m = np.load(os.path.join(data_dir, "meta.npz"))
        self.state, self.actions = m["state"], m["actions"]
        self.ep_id, self.t, self.task_id = m["ep_id"], m["t"], m["task_id"]
        self.ep_start, self.ep_len = m["ep_start"], m["ep_len"]
        self.tasks = json.load(open(os.path.join(data_dir, "tasks.json")))
        self.stats = json.load(open(os.path.join(data_dir, "stats.json")))
        self.norm = Normalizer(self.stats)
        self.norm_actions = self.norm.norm_actions(self.actions)
        self.norm_state = self.norm.norm_state(self.state)
        n = len(self.t)
        assert self.frames.shape[0] == n
        # frames of an episode are contiguous and in time order
        assert (np.arange(n) == self.ep_start[self.ep_id] + self.t).all()

    def __len__(self):
        return len(self.t)

    def chunk_indices(self, i: int):
        """Indices of a_t..a_{t+H-1} and the padding mask (True = beyond the episode end)."""
        ep, t = self.ep_id[i], self.t[i]
        L = self.ep_len[ep]
        ks = np.arange(self.chunk)
        is_pad = t + ks >= L
        idx = np.where(is_pad, i + (L - 1 - t), i + ks)  # pad slots point at the last real step (masked anyway)
        return idx, is_pad

    def __getitem__(self, i: int):
        idx, is_pad = self.chunk_indices(i)
        actions = self.norm_actions[idx].copy()
        actions[is_pad] = 0.0
        return {
            "images": torch.from_numpy(np.ascontiguousarray(self.frames[i])),  # uint8 [2,256,256,3]
            "state": torch.from_numpy(self.norm_state[i]),
            "actions": torch.from_numpy(actions),  # [H,7] normalised
            "is_pad": torch.from_numpy(is_pad),  # [H] bool
            "task_id": int(self.task_id[i]),
        }


class InfiniteRandomSampler(torch.utils.data.Sampler):
    """Each rank draws an independent permutation stream (seed + rank); no epoch bookkeeping needed."""

    def __init__(self, n: int, seed: int, rank: int):
        self.n, self.seed, self.rank = n, seed, rank

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed * 1000 + self.rank)
        while True:
            yield from torch.randperm(self.n, generator=g).tolist()
