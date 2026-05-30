#!/bin/bash


datasets=(
  Epilepsy
  Shoaib
  EMGPhysical
  SelfRegulationSCP1
  WESADchest
  PAMAP2
)

for d in "${datasets[@]}"; do
  echo "Running Main.py for $d"

  python Main.py --dataset_name "$d" &&
  python data_logger.py "$d"

  echo "Finished $d. Sleeping 300 seconds..."
  sleep 300
done