#!/bin/bash

# Activate the conda environment
source /home/zjy/miniconda3/etc/profile.d/conda.sh
conda activate BodyPressure

num_threads=4

python3 ../1_prepare_dataset.py \
    --random_state 42 \
    --test_size 100 \
    --num_threads $num_threads \
