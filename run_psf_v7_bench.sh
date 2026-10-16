#!/bin/bash
# v7 = v6 with a SHORT backup (15 steps) + neural terminal certificate (models/terminal_barrier_8d.pth, h >= 0.2)
# mu=1.0, v=6, K=50 and K=1000, 30 standard seeds (sharded).
cd /home/raj/agile_nicbf
PY=/home/raj/miniconda3/envs/carla_env/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PSF='{"backup": true, "horizon": 20, "carry": true, "max_heading": 1.4, "backup_yaw_gain": 0.1, "a_brake": 4.5, "backup_steps": 15, "terminal_net_path": "/home/raj/agile_nicbf/models/terminal_barrier_8d.pth", "terminal_thresh": 0.2}'
for K in 50 1000; do
  for i in 0 1 2 3 4 5; do
    $PY stress_grid.py --mu 1.0 --speeds 6 --Ks $K --episodes 30 --methods predictive_filter \
        --psf "$PSF" --shard $i/6 --out results/psf_v7_k$K.jsonl > logs/psf_v7_k${K}_$i.log 2>&1 &
  done
done
wait
echo done
