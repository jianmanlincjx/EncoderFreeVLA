"""Render task 0 / init 0 of a suite with the LIBERO config in $LIBERO_CONFIG_PATH and report mean RGB."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")
from efvla.eval_libero import DUMMY, NUM_STEPS_WAIT, make_env, obs_to_inputs  # noqa: E402

from libero.libero import get_libero_path  # noqa: E402

env, task, inits = make_env(sys.argv[1] if len(sys.argv) > 1 else "libero_object", 0)
env.reset()
obs = env.set_init_state(inits[0])
for _ in range(NUM_STEPS_WAIT):
    obs, _, _, _ = env.step(DUMMY)
imgs, _ = obs_to_inputs(obs)
print("assets", get_libero_path("assets"))
print("agentview mean RGB", imgs[0].reshape(-1, 3).mean(0).round(1), " wrist", imgs[1].reshape(-1, 3).mean(0).round(1))
if len(sys.argv) > 2:
    from PIL import Image

    Image.fromarray(imgs[0]).save(sys.argv[2])
