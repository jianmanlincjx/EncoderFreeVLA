"""Closed-loop LIBERO evaluation (openpi/OpenVLA-OFT protocol).
  python -m efvla.eval_libero --ckpt <ckpt>/model.pt --suite libero_object --tasks 0,1 --episodes 50 --out <dir>
Protocol: official 50 init states per task, env.seed(7), reset -> set_init_state, 10 dummy steps
[0,0,0,0,0,0,-1], then up to max_steps policy steps; a chunk of H actions is executed fully before replanning.
Observations: images = raw[::-1, ::-1] (matches the openpi dataset), state8 = eef_pos + axis-angle + gripper_qpos.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}
DUMMY = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
NUM_STEPS_WAIT = 10


def obs_to_inputs(obs):
    import robosuite.utils.transform_utils as TU

    imgs = np.stack([obs["agentview_image"][::-1, ::-1], obs["robot0_eye_in_hand_image"][::-1, ::-1]], 0)
    state = np.concatenate([obs["robot0_eef_pos"], TU.quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"]]).astype(np.float32)
    return np.ascontiguousarray(imgs), state


def make_env(suite: str, tid: int, seed: int = 7):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bm = benchmark.get_benchmark_dict()[suite]()
    task = bm.get_task(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    inits = torch.load(os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file),
                       weights_only=False)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(seed)
    return env, task, inits


def run_task(policy, suite, tid, episodes, out_dir, video_every=0, seed=7):
    env, task, inits = make_env(suite, tid, seed)
    lang = task.language
    assert lang in policy.tasks, f"env language not in training tasks: {lang!r}"
    max_steps = MAX_STEPS[suite]
    results = []
    for ep in range(episodes):
        t0 = time.time()
        policy.reseed(seed * 100000 + tid * 1000 + ep)
        env.reset()
        obs = env.set_init_state(inits[ep % len(inits)])
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(DUMMY)
        success, t, queue, frames = False, 0, [], []
        while t < max_steps:
            imgs, state = obs_to_inputs(obs)
            if video_every and ep % video_every == 0:
                frames.append(np.concatenate([imgs[0], imgs[1]], 1))
            if not queue:
                queue = list(policy.act(imgs, state, lang))
            obs, _, done, _ = env.step(queue.pop(0).tolist())
            t += 1
            if done:
                success = True
                break
        results.append({"episode": ep, "success": success, "steps": t, "sec": round(time.time() - t0, 1)})
        print(f"[eval] task {tid} ep {ep} success={success} steps={t} {time.time() - t0:.0f}s", flush=True)
        if frames:
            import imageio

            os.makedirs(os.path.join(out_dir, "videos"), exist_ok=True)
            imageio.mimsave(os.path.join(out_dir, "videos", f"task{tid:02d}_ep{ep:02d}_{'ok' if success else 'fail'}.mp4"),
                            frames, fps=20)
    env.close()
    rec = {"suite": suite, "task_id": tid, "language": lang, "episodes": len(results),
           "successes": sum(r["success"] for r in results), "results": results,
           "ckpt_step": policy.step, "blank_images": policy.blank}
    tmp = os.path.join(out_dir, f"task_{tid:02d}.json.tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=2)
    os.replace(tmp, os.path.join(out_dir, f"task_{tid:02d}.json"))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blank_images", action="store_true")
    ap.add_argument("--video_every", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.makedirs(a.out, exist_ok=True)
    from efvla.policy import Policy

    policy = Policy(a.ckpt, "cuda", blank_images=a.blank_images)
    for tid in [int(x) for x in a.tasks.split(",")]:
        if os.path.exists(os.path.join(a.out, f"task_{tid:02d}.json")):
            print(f"[eval] skip task {tid} (done)", flush=True)
            continue
        r = run_task(policy, a.suite, tid, a.episodes, a.out, a.video_every, a.seed)
        print(f"[eval] TASK_DONE {tid} {r['successes']}/{r['episodes']}", flush=True)


if __name__ == "__main__":
    main()
