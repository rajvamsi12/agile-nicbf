#!/bin/bash
# Predictive filter v2 (MPPI samples + robust progress selection) and v3 (v2 + brake-to-stop backup),
# mu=1.0, v=6, K=50, 30 standard seeds (sharded). Filter runs on CPU.
cd /home/raj/agile_nicbf
PY=/home/raj/miniconda3/envs/carla_env/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
C="--mu 1.0 --speeds 6 --Ks 50 --episodes 30 --methods predictive_filter"
for i in 0 1 2 3 4 5; do
  $PY stress_grid.py $C --psf '{}' --shard $i/6 --out results/psf_v2.jsonl > logs/psf_v2_$i.log 2>&1 &
  $PY stress_grid.py $C --psf '{"backup": true, "horizon": 20}' --shard $i/6 --out results/psf_v3_backup.jsonl > logs/psf_v3_$i.log 2>&1 &
done
wait
echo done
