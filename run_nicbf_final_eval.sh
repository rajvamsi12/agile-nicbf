#!/bin/bash
# Stage 5 final evaluation: Neural ICBF pipeline (nicbf = pi_theta nominal, paper-faithful; nicbf_mppi = MPPI nominal)
# at v_target 3.5 and 4.0 m/s, mu = 1.0, 30 standard seeds, frozen config prototype_neural_ode/dynamic/nicbf_final_config.json.
cd /home/raj/agile_nicbf
PY=/home/raj/miniconda3/envs/carla_env/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
CFG=$(cat prototype_neural_ode/dynamic/nicbf_final_config.json)
echo "===== STAGE 5b: final NICBF evaluation  $(date)  config $CFG" >> logs/nicbf_overnight.log
for i in 0 1 2 3 4 5; do
  $PY stress_grid.py --mu 1.0 --speeds 3.5 4.0 --Ks 50 --episodes 30 --methods nicbf nicbf_mppi \
      --nicbf "$CFG" --shard $i/6 --out results/nicbf_eval_pipeline.jsonl > logs/nicbf_final_$i.log 2>&1 &
done
wait
echo "final evaluation done $(date)" >> logs/nicbf_overnight.log
