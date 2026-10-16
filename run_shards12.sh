#!/bin/bash
cd /home/raj/agile_nicbf
/home/raj/miniconda3/envs/carla_env/bin/python stress_grid.py --mu 1.0 0.7 --speeds 6 8 --Ks 50 --episodes 30 --methods mppi_alone mppi_oracle_mu analytical_hocbf agile_nicbf_coupled --shard 1/3 --out results/ablate_clamp_on.jsonl > shard1_on.log 2>&1 &
/home/raj/miniconda3/envs/carla_env/bin/python stress_grid.py --mu 1.0 0.7 --speeds 6 8 --Ks 50 --episodes 30 --methods mppi_alone mppi_oracle_mu analytical_hocbf agile_nicbf_coupled --shard 2/3 --out results/ablate_clamp_on.jsonl > shard2_on.log 2>&1 &
wait
