#!/bin/bash
# v10 (neural screen + exact verify), v10x (exact screen, no network), v10x_mu95 (v10x assuming 5% less grip).
# All: numpy backend, 0.10 m commitment margin, carry-over, emergency yaw-damping brake when nothing is safe.
# mu=1.0, v=6, K=50 and K=1000, 30 standard seeds, 4 shards per run. Configs in diag/cfg_*.json.
cd /home/raj/agile_nicbf
PY=/home/raj/miniconda3/envs/carla_env/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for name in v10 v10x v10x_mu95; do
  PSF=$(cat diag/cfg_$name.json)
  for K in 50 1000; do
    for i in 0 1 2 3; do
      $PY stress_grid.py --mu 1.0 --speeds 6 --Ks $K --episodes 30 --methods predictive_filter \
          --psf "$PSF" --shard $i/4 --out results/psf_${name}_k$K.jsonl > logs/psf_${name}_k${K}_$i.log 2>&1 &
    done
  done
done
wait
echo done
