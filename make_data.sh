#!/bin/bash

mkdir -p with_selection
mkdir -p without_selection

############################
# Memory without selection #
############################
nvidia-smi \
    --query-gpu=timestamp,index,memory.total,memory.reserved,memory.free,memory.used \
    --format=csv,nounits \
    --loop-ms=10 \
    --filename=without_selection/gpu_memory_profile.csv &
NVIDIA_SMI_PID=$!

../../run_datagen.sh reconstruct . 10 \
    --digi=$DIGI \
    --overwrite_config="{ \"embeddingDim\": ${EMBEDDING_DIM} }"

kill -INT $NVIDIA_SMI_PID


#########################
# Memory with selection #
#########################
nvidia-smi \
    --query-gpu=timestamp,index,memory.total,memory.reserved,memory.free,memory.used \
    --format=csv,nounits \
    --loop-ms=10 \
    --filename=with_selection/gpu_memory_profile.csv &
NVIDIA_SMI_PID=$!

../../run_datagen.sh reconstruct . 10 \
    --digi=$DIGI \
    --select \
    --overwrite_config="{ \"embeddingDim\": ${EMBEDDING_DIM} }"

kill -INT $NVIDIA_SMI_PID


###################################
# Data & timing without selection #
###################################
../../run_datagen.sh reconstruct . 10 \
    --digi=$DIGI \
    --overwrite_config="{ \"embeddingDim\": ${EMBEDDING_DIM} }" \
    --with_ckf \
    --with_truthtracking

mv *.tsv *.root without_selection/


################################
# Data & timing with selection #
################################
../../run_datagen.sh reconstruct . 10 \
    --digi=$DIGI \
    --select \
    --overwrite_config="{ \"embeddingDim\": ${EMBEDDING_DIM} }" \
    --with_ckf \
    --with_truthtracking

mv *.tsv *.root with_selection/
