#!/bin/bash

DIGI=$1
EMBEDDING_DIM=$2

mkdir -p with_selection
mkdir -p without_selection

function profile_memory {

    nvidia-smi \
        --query-gpu=timestamp,index,memory.total,memory.reserved,memory.free,memory.used \
        --format=csv,nounits \
        --loop-ms=10 \
        --filename=gpu_memory_profile.csv &
    NVIDIA_SMI_PID=$!

    $HOME/exatrkx/run_datagen.sh reconstruct . 10 \
        --digi=$DIGI \
        --overwrite_config="{ \"embeddingDim\": ${EMBEDDING_DIM} }" \
        $@

    kill -INT $NVIDIA_SMI_PID

}

function data_and_timing {

    $HOME/exatrkx/run_datagen.sh reconstruct . 10 \
        --digi=$DIGI \
        --overwrite_config="{ \"embeddingDim\": ${EMBEDDING_DIM} }" \
        --with_ckf \
        --with_truthtracking \
        $@

}

# Without selection
(
    mkdir -p without_selection;
    cd without_selection;
    ln -sf ../torchscript
    profile_memory;
    data_and_timing;
)

# With selection
(
    mkdir -p without_selection;
    cd without_selection;
    ln -sf ../torchscript
    profile_memory --select;
    data_and_timing --select;
)
