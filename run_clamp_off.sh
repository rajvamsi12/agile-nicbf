#!/bin/bash
cd /home/raj/agile_nicbf
/home/raj/miniconda3/envs/carla_env/bin/python stress_grid.py --mu 1.0 0.7 --speeds 6 8 --Ks 50 --episodes 30 --methods analytical_hocbf agile_nicbf_coupled --no-brake-clamp --shard 0/3 --out results/ablate_clamp_off.jsonl > shard0_off.log 2>&1 &
/home/raj/miniconda3/envs/carla_env/bin/python stress_grid.py --mu 1.0 0.7 --speeds 6 8 --Ks 50 --episodes 30 --methods analytical_hocbf agile_nicbf_coupled --no-brake-clamp --shard 1/3 --out results/ablate_clamp_off.jsonl > shard1_off.log 2>&1 &
/home/raj/miniconda3/envs/carla_env/bin/python stress_grid.py --mu 1.0 0.7 --speeds 6 8 --Ks 50 --episodes 30 --methods analytical_hocbf agile_nicbf_coupled --no-brake-clamp --shard 2/3 --out results/ablate_clamp_off.jsonl > shard2_off.log 2>&1 &
wait
echo 'All clamp-off shards completed!'
