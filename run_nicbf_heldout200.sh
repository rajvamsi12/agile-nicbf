#!/bin/bash
# Held-out evaluation: 200 fresh seeds (index 100..299, seed 3801..11164; none used for tuning or earlier runs),
# v_target 3.5 and 4.0 m/s, mu = 1.0, K = 50, CPU only. Frozen config prototype_neural_ode/dynamic/nicbf_final_config.json.
cd "$(dirname "$0")"
PY=${PY:-/home/raj/miniconda3/envs/carla_env/bin/python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
CFG=$(cat prototype_neural_ode/dynamic/nicbf_final_config.json)
N=${N:-30}
echo "===== HELD-OUT 200-seed evaluation  $(date)  config $CFG" >> logs/nicbf_overnight.log
for i in $(seq 0 $((N-1))); do
  $PY stress_grid.py --mu 1.0 --speeds 3.5 4.0 --Ks 50 --episodes 200 --seed-start 100 --device cpu \
      --methods mppi_alone analytical_hocbf nicbf nicbf_mppi --nicbf "$CFG" \
      --shard $i/$N --out results/nicbf_heldout200.jsonl > logs/nicbf_heldout200_$i.log 2>&1 &
done
wait
echo "held-out evaluation done $(date)" >> logs/nicbf_overnight.log
