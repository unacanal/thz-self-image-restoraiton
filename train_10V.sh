#!/bin/bash

: ' ===== Experiment 1: One-to-One ===== '
echo "Exp1: One-to-One"
volt='10V'
for i in $(seq 0.1 0.1 3.0); do
    for j in $(seq 0.1 0.1 3.0); do
            echo $i $j
            CUDA_VISIBLE_DEVICES=0 python train.py \
            --img "hyperspectral/${volt}_30ps_e/${volt}_30ps_e_${i}THz.png" \
            --target "hyperspectral/${volt}_30ps_e/${volt}_30ps_e_${j}THz.png" \
            --exp o2o_${volt} \
            --volt ${volt}
            CUDA_VISIBLE_DEVICES=0 python test.py \
            --ckpt_path checkpoints_volt/ZSSR_o2o_${volt}/${volt}_30ps_e_${i}_to_${j}THz_latest.pt \
            --save_dir o2o_${volt} \
            --volt ${volt}
    done
done
