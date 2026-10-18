"""Build a uint8 frame cache + normalisation stats for one LIBERO suite from the openpi-format dataset.

Source: /pfs/pfs-Tr4Uts/libero (LeRobot v2.1, parquet with embedded PNG; `image`/`wrist_image` are 256x256 and
already rotated 180 deg relative to raw LIBERO renders; `state` = eef_pos(3)+axis-angle(3)+gripper_qpos(2);
`actions` = 7-d delta EEF + gripper in {-1, +1}).

Outputs in cache_dir (all written atomically):
  frames.npy   uint8 [N, 2, 256, 256, 3]   (cam 0 = image/agentview, cam 1 = wrist_image)
  meta.npz     state[N,8] float32, actions[N,7] float32, ep_id[N], t[N], task_id[N], ep_start[E], ep_len[E],
               src_episode[E]
  tasks.json   list of task strings (index = task_id), in benchmark order
  stats.json   per-dim q01/q99/mean/std/min/max for state and actions
Usage: python -m efvla.data.libero_cache --suite libero_object --out <dir> [--workers 32]
"""
import argparse
import io
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image


def suite_task_languages(suite: str) -> list[str]:
    from libero.libero import benchmark

    bm = benchmark.get_benchmark_dict()[suite]()
    return [bm.get_task(i).language for i in range(bm.n_tasks)]


def _decode(cell) -> np.ndarray:
    if isinstance(cell, dict):
        img = Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
    else:
        img = Image.fromarray(np.asarray(cell))
    arr = np.asarray(img, dtype=np.uint8)
    assert arr.shape == (256, 256, 3), arr.shape
    return arr


def _load_episode(args):
    root, ep = args
    path = f"{root}/data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet"
    df = pd.read_parquet(path)
    fi = df["frame_index"].to_numpy()
    assert (fi == np.arange(len(df))).all(), f"episode {ep}: frame_index not 0..L-1"
    assert (df["episode_index"].to_numpy() == ep).all()
    frames = np.stack(
        [np.stack([_decode(a), _decode(w)], 0) for a, w in zip(df["image"], df["wrist_image"])], 0
    )
    state = np.stack(df["state"].to_numpy()).astype(np.float32)
    actions = np.stack(df["actions"].to_numpy()).astype(np.float32)
    assert state.shape == (len(df), 8) and actions.shape == (len(df), 7)
    return ep, frames, state, actions


def _atomic_save_npz(path, **arrays):
    tmp = path + f".tmp{os.getpid()}.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _atomic_write_json(path, obj):
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def compute_stats(x: np.ndarray) -> dict:
    return {
        "q01": np.quantile(x, 0.01, axis=0).tolist(),
        "q99": np.quantile(x, 0.99, axis=0).tolist(),
        "mean": x.mean(0).tolist(),
        "std": x.std(0).tolist(),
        "min": x.min(0).tolist(),
        "max": x.max(0).tolist(),
    }


def build(root: str, suite: str, out: str, workers: int):
    os.makedirs(out, exist_ok=True)
    tasks = suite_task_languages(suite)
    assert len(tasks) == len(set(tasks)) == 10, tasks
    task_index = {t: i for i, t in enumerate(tasks)}
    eps = [json.loads(l) for l in open(f"{root}/meta/episodes.jsonl")]
    sel = []
    for e in eps:
        assert len(e["tasks"]) == 1, e
        if e["tasks"][0] in task_index:  # exact string match, never keyword match
            sel.append((e["episode_index"], task_index[e["tasks"][0]], e["length"]))
    found = {t for _, t, _ in sel}
    assert found == set(range(len(tasks))), f"missing tasks: {set(range(len(tasks))) - found}"
    sel.sort()
    n = sum(l for _, _, l in sel)
    print(f"[cache] suite={suite} episodes={len(sel)} frames={n}", flush=True)

    frames_tmp = os.path.join(out, f"frames.npy.tmp{os.getpid()}.npy")
    frames = np.lib.format.open_memmap(frames_tmp, mode="w+", dtype=np.uint8, shape=(n, 2, 256, 256, 3))
    state = np.zeros((n, 8), np.float32)
    actions = np.zeros((n, 7), np.float32)
    ep_id = np.zeros(n, np.int32)
    t_idx = np.zeros(n, np.int32)
    task_id = np.zeros(n, np.int32)
    ep_start = np.zeros(len(sel), np.int64)
    ep_len = np.zeros(len(sel), np.int32)
    src_ep = np.array([s[0] for s in sel], np.int32)
    offsets, off = {}, 0
    for k, (ep, tid, L) in enumerate(sel):
        offsets[ep] = (k, off, tid, L)
        ep_start[k], ep_len[k] = off, L
        off += L

    t0 = time.time()
    done = 0
    with Pool(workers) as pool:
        for ep, fr, st, ac in pool.imap_unordered(_load_episode, [(root, s[0]) for s in sel]):
            k, o, tid, L = offsets[ep]
            assert len(fr) == L, (ep, len(fr), L)
            frames[o : o + L] = fr
            state[o : o + L] = st
            actions[o : o + L] = ac
            ep_id[o : o + L] = k
            t_idx[o : o + L] = np.arange(L)
            task_id[o : o + L] = tid
            done += 1
            if done % 50 == 0:
                print(f"[cache] {done}/{len(sel)} episodes, {time.time() - t0:.0f}s", flush=True)
    frames.flush()
    del frames
    os.replace(frames_tmp, os.path.join(out, "frames.npy"))

    grip = np.unique(actions[:, 6])
    assert set(grip.tolist()) <= {-1.0, 1.0}, f"gripper action values {grip}"
    _atomic_save_npz(
        os.path.join(out, "meta.npz"),
        state=state, actions=actions, ep_id=ep_id, t=t_idx, task_id=task_id,
        ep_start=ep_start, ep_len=ep_len, src_episode=src_ep,
    )
    _atomic_write_json(os.path.join(out, "tasks.json"), tasks)
    stats = {
        "state": compute_stats(state),
        "actions": compute_stats(actions),
        "n_frames": int(n),
        "n_episodes": len(sel),
        "suite": suite,
        "source": root,
    }
    _atomic_write_json(os.path.join(out, "stats.json"), stats)
    print(f"[cache] DONE frames={n} episodes={len(sel)} gripper_values={grip.tolist()} {time.time() - t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/pfs/pfs-Tr4Uts/libero")
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    build(a.root, a.suite, a.out, a.workers)
