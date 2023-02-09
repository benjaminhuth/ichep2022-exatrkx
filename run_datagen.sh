#!/bin/bash

# Set environment variables etc. here...

# Truth-digi runs
python3 datagen.py ./config/truth ./inference_results/truth --emb=8 --profile_gpu
python3 datagen.py ./config/truth ./inference_results/truth --emb=8 --with_ckf --with_truthtracking

# Smeared-digi runs
python3 datagen.py ./config/smeared ./inference_results/smeared --emb=12 --profile_gpu
python3 datagen.py ./config/smeared ./inference_results/smeared --emb=12 --with_ckf --with_truthtracking
