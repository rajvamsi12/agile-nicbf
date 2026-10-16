#!/bin/bash
# Benchmark the multi-step predictive safety filter at mu=1.0, v=6, K=50, 30 standard seeds (sharded).
cd /home/raj/agile_nicbf
PY=/home/raj/miniconda3/envs/carla_env/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for i in 0 1 2 3 4 5; do
  $PY stress_grid.py --mu 1.0 --speeds 6 --Ks 50 --episodes 30 --methods predictive_filter \
      --shard $i/6 --out results/predictive_filter.jsonl > logs/psf_$i.log 2>&1 &
done
wait
echo done
