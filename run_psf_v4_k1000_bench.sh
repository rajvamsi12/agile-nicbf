#!/bin/bash
# v4 = backup + carry-over (K=50) and v2 at K=1000; mu=1.0, v=6, 30 standard seeds (sharded).
cd /home/raj/agile_nicbf
PY=/home/raj/miniconda3/envs/carla_env/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for i in 0 1 2 3 4 5; do
  $PY stress_grid.py --mu 1.0 --speeds 6 --Ks 50 --episodes 30 --methods predictive_filter \
      --psf '{"backup": true, "horizon": 20, "carry": true}' --shard $i/6 --out results/psf_v4_backup_carry.jsonl > logs/psf_v4_$i.log 2>&1 &
  $PY stress_grid.py --mu 1.0 --speeds 6 --Ks 1000 --episodes 30 --methods predictive_filter \
      --psf '{}' --shard $i/6 --out results/psf_v2_k1000.jsonl > logs/psf_v2k1000_$i.log 2>&1 &
done
wait
echo done
