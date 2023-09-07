#!/bin/bash

: ' ===== Experiment 1: One-to-One ===== '
echo "Exp1: One-to-One"
volt='40V'
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


: ' Recurrent '
# 1.2 to 1.2(0.8to1.2)
CUDA_VISIBLE_DEVICES=7 python train.py --img hyperspectral/40V_30ps_e/40V_30ps_e_1.2THz.png --target 'results_volt/o2o_40V/1.2/40V_30ps_e_1.2THz(0.8to1.2).png'
CUDA_VISIBLE_DEVICES=7 python test.py --test_img 'results_volt/o2o_40V/1.2/40V_30ps_e_1.2THz(0.8to1.2).png' --ckpt_path 'checkpoints/ZSSR_o2o/40V_30ps_e_1.2_to_1.2THz(0.8to1.2)_latest.pt' --save_dir recurrent --volt 40V


# 0.8 to 1.2(0.8to1.2)
CUDA_VISIBLE_DEVICES=7 python train.py --img hyperspectral/40V_30ps_e/40V_30ps_e_0.8THz.png --target 'results_volt/o2o_40V/1.2/40V_30ps_e_1.2THz(0.8to1.2).png'
CUDA_VISIBLE_DEVICES=7 python test.py --test_img 'results_volt/o2o_40V/1.2/40V_30ps_e_1.2THz(0.8to1.2).png' --ckpt_path 'checkpoints/ZSSR_o2o/40V_30ps_e_0.8_to_1.2THz(0.8to1.2)_latest.pt' --save_dir recurrent --volt 40V 