#!/bin/bash

# One run to get all data
../../run_datagen.sh reconstruct . 10 --digi=truth --select --overwrite_config='{ "embeddingDim": 8 }' --with_ckf --with_truthtracking
mv timing.tsv timing_with_ckf_and_truthtracking.tsv
