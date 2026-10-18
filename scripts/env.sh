# source me: common environment for training / evaluation on fm1
export EFVLA=/pfs/pfs-Tr4Uts/JM_work/encoder_free_vla
export PY=/pfs/pfs-Tr4Uts/JM_work/venvs/molmoact2_lerobot_venv/bin/python
export TORCHRUN=/pfs/pfs-Tr4Uts/JM_work/venvs/molmoact2_lerobot_venv/bin/torchrun
export LIBERO_CONFIG_PATH=$EFVLA/libero_config   # original LIBERO assets bundled with the venv (not LIBERO-plus)
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$EFVLA
