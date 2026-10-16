#!/bin/bash
# Benchmark the wall steering-authority guard at mu=1.0, v=6, K=50, 30 standard seeds (sharded).
cd /home/raj/agile_nicbf
PY=/home/raj/miniconda3/envs/carla_env/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs
COMMON="--mu 1.0 --speeds 6 --Ks 50 --episodes 30"
for i in 0 1 2 3 4 5; do
  $PY stress_grid.py $COMMON --methods analytical_hocbf neural_icbf agile_nicbf_coupled \
      --wall-steer-guard drop_row --shard $i/6 --out results/wallguard_drop_row.jsonl > logs/wg_drop_row_$i.log 2>&1 &
done
for i in 0 1 2; do
  $PY stress_grid.py $COMMON --methods analytical_hocbf \
      --wall-steer-guard zero_steer --shard $i/3 --out results/wallguard_zero_steer.jsonl > logs/wg_zero_steer_$i.log 2>&1 &
done
wait
echo done
