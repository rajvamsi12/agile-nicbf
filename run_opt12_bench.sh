#!/bin/bash
# Benchmark drop_row + obstacle steer guard + slip brake limit at mu=1.0, v=6, K=50, 30 seeds (sharded).
cd /home/raj/agile_nicbf
PY=/home/raj/miniconda3/envs/carla_env/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for i in 0 1 2 3 4 5; do
  $PY stress_grid.py --mu 1.0 --speeds 6 --Ks 50 --episodes 30 \
      --methods analytical_hocbf neural_icbf agile_nicbf_coupled \
      --wall-steer-guard drop_row --obs-steer-guard --slip-brake-limit 1.0 \
      --shard $i/6 --out results/opt12_guard.jsonl > logs/opt12_$i.log 2>&1 &
done
wait
echo done
