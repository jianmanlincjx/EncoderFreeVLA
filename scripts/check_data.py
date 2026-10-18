"""Data/env convention checks (needs EGL; run with one free GPU visible):
  python scripts/check_data.py --cache data_cache/libero_object --out logs/check_data
1. env task languages == cache tasks.json (exact strings, same order)
2. image orientation: env frame [::-1, ::-1] vs dataset frame 0 of the same task (mean abs diff, also unflipped)
3. state8 at reset vs dataset frame-0 states of the same task
4. gripper convention: +1 closes (finger qpos shrinks), -1 opens
Writes side-by-side PNGs for visual inspection.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="0,5")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from PIL import Image
    from efvla.data.libero_cache import suite_task_languages
    from efvla.eval_libero import DUMMY, NUM_STEPS_WAIT, make_env, obs_to_inputs

    tasks = json.load(open(os.path.join(a.cache, "tasks.json")))
    env_tasks = suite_task_languages(a.suite)
    assert tasks == env_tasks, (tasks, env_tasks)
    print(f"[1] languages identical ({len(tasks)} tasks)")
    frames = np.load(os.path.join(a.cache, "frames.npy"), mmap_mode="r")
    m = np.load(os.path.join(a.cache, "meta.npz"))
    report = {}
    for tid in [int(x) for x in a.tasks.split(",")]:
        env, task, inits = make_env(a.suite, tid)
        env.reset()
        obs = env.set_init_state(inits[0])
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(DUMMY)
        imgs, state = obs_to_inputs(obs)
        raw_agent = obs["agentview_image"]
        first = np.where((m["task_id"] == tid) & (m["t"] == 0))[0]
        ds0 = np.asarray(frames[first[0]])
        d_flip = float(np.abs(imgs[0].astype(float) - ds0[0].astype(float)).mean())
        d_raw = float(np.abs(raw_agent.astype(float) - ds0[0].astype(float)).mean())
        d_wflip = float(np.abs(imgs[1].astype(float) - ds0[1].astype(float)).mean())
        d_wraw = float(np.abs(obs["robot0_eye_in_hand_image"].astype(float) - ds0[1].astype(float)).mean())
        Image.fromarray(np.concatenate([np.concatenate([imgs[0], ds0[0]], 1), np.concatenate([imgs[1], ds0[1]], 1)], 0)
                        ).save(os.path.join(a.out, f"task{tid:02d}_env_vs_dataset.png"))
        s0 = m["state"][first]
        diff = np.abs(state - s0.mean(0))
        spread = s0.std(0) + 1e-6
        # gripper convention
        o = obs
        q_open = o["robot0_gripper_qpos"].copy()
        for _ in range(20):
            o, _, _, _ = env.step([0, 0, 0, 0, 0, 0, 1.0])
        q_closed = o["robot0_gripper_qpos"].copy()
        for _ in range(20):
            o, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1.0])
        q_reopen = o["robot0_gripper_qpos"].copy()
        env.close()
        report[tid] = {
            "agentview mean|diff| flipped vs raw": [round(d_flip, 1), round(d_raw, 1)],
            "wrist mean|diff| flipped vs raw": [round(d_wflip, 1), round(d_wraw, 1)],
            "env state8": np.round(state, 4).tolist(),
            "dataset t=0 state mean": np.round(s0.mean(0), 4).tolist(),
            "|env - mean| / std": np.round(diff / spread, 2).tolist(),
            "gripper qpos open/closed(+1)/reopen(-1)": [np.round(q_open, 4).tolist(), np.round(q_closed, 4).tolist(),
                                                        np.round(q_reopen, 4).tolist()],
        }
        ok_orient = d_flip < d_raw and d_wflip < d_wraw
        ok_grip = abs(q_closed[0] - q_closed[1]) < abs(q_open[0] - q_open[1]) and \
            abs(q_reopen[0] - q_reopen[1]) > abs(q_closed[0] - q_closed[1])
        print(f"[task {tid}] orientation {'OK' if ok_orient else 'FAIL'}  gripper(+1 closes) {'OK' if ok_grip else 'FAIL'}")
        print(json.dumps(report[tid], indent=1))
    json.dump(report, open(os.path.join(a.out, "check_data.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
