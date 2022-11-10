#!/bin/bash

# One run to get all data
../../run_datagen.sh reconstruct . 10 --digi=smear --select --overwrite_config='{ "embeddingDim": 12 }' --with_ckf --with_truthtracking
mv timing.tsv timing_with_ckf_and_truthtracking.tsv

# One run to record GPU memory
echo "Monitor GPU memory..."
(nvidia-smi --query-gpu=timestamp,index,memory.total,memory.reserved,memory.free,memory.used --format=csv,nounits --loop-ms=10 --filename=gpu_memory_profile.csv) &
NVIDIA_SMI_PID=$!

../../run_datagen.sh reconstruct . 10 --digi=smear --select --overwrite_config='{ "embeddingDim": 12 }'

kill -INT $NVIDIA_SMI_PID
