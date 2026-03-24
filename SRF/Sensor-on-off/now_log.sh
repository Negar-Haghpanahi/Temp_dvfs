#!/bin/bash

# Start log script in background
sleep 1
sh print_log_data.sh "$1" &

# Capture its PID
LOGPROGRAMPID=$!

sleep 1

# Loop (currently runs once because '1' is fixed)
for i in 1
do
    echo "$i"
    python Main.py \
  --dataset_name Epilepsy \
  --num_exits 3 \
  --proportions 0.37 0.55 1 \
  --th_combination 0.94 1.2 1.22
done

# Kill log script when exiting
trap 'kill -15 $LOGPROGRAMPID' EXIT


                  
