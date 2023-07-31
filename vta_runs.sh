#!/bin/bash

for c in $(seq 10 10 90)
do
    echo "Running compression coefficient $c"
    python vta_maze2d_segmentation_rl_test.py model.compression_coeff=$c
done