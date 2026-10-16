#!/bin/bash
# v6 = backup + carry + heading limit (80 deg) + yaw-damping backup; mu=1.0, v=6, K=50 and K=1000, 30 seeds.
cd /home/raj/agile_nicbf
PY=/home/raj/miniconda3/envs/carla_env/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PSF='{"backup": true, "horizon": 20, "carry": true, "max_heading": 1.4, "backup_yaw_gain": 0.1, "a_brake": 4.5}'
for K in 50 1000; do
  for i in 0 1 2 3 4 5; do
    $PY stress_grid.py --mu 1.0 --speeds 6 --Ks $K --episodes 30 --methods predictive_filter \
        --psf "$PSF" --shard $i/6 --out results/psf_v6_k$K.jsonl > logs/psf_v6_k${K}_$i.log 2>&1 &
  done
done
wait
echo done
