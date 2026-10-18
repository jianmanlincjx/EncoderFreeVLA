#!/bin/bash
# build LIBERO-Object uint8 cache (CPU only)
P=/pfs/pfs-Tr4Uts/JM_work/encoder_free_vla
PY=/pfs/pfs-Tr4Uts/JM_work/venvs/molmoact2_lerobot_venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $P
mkdir -p logs data_cache
$PY -m efvla.data.libero_cache --suite libero_object --out $P/data_cache/libero_object --workers 32 > logs/cache_libero_object.log 2>&1
echo "EXIT $?" >> logs/cache_libero_object.log
