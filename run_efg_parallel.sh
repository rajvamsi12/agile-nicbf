#!/bin/bash
# Run Parts E/F/G of run_efg_diagnostics.py as N parallel seed-shards each, then print the tables.
# Usage: ./run_efg_parallel.sh [N=6] [parts="E F G"]
cd /home/raj/agile_nicbf
PY=/home/raj/miniconda3/envs/carla_env/bin/python
N=${1:-6}
PARTS=${2:-"E F G"}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1   # one core per worker, avoid oversubscription
mkdir -p logs
for part in $PARTS; do
  for i in $(seq 0 $((N-1))); do
    $PY -u run_efg_diagnostics.py --part $part --shard $i/$N > logs/efg_${part}_shard$i.log 2>&1 &
  done
done
wait
# Unsharded pass: every episode is already done, so this only loads results and prints the tables.
for part in $PARTS; do
  $PY -u run_efg_diagnostics.py --part $part > logs/efg_${part}_report.log 2>&1
done
echo "done; tables in logs/efg_*_report.log"
