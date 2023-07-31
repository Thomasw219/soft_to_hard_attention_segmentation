#!/bin/bash

for c in $(seq 0.12 0.02 0.28)
do
    echo "Running compression coefficient $c"
    python maze2d_segmentation_rl_test.py model.time_loss_weight=$c
done