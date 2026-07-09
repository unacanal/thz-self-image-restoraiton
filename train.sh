#!/bin/bash

: ' ===== Experiment 1: One-to-One ===== '
# echo "Exp1: One-to-One"
# function float_lt() {
#     awk -v a="$1" -v b="$2" 'BEGIN {print (a < b)}'
# }
# for i in $(seq 0.1 0.1 3.0); do
#     for j in $(seq 0.1 0.1 3.0); do
#         # if [[ $(float_lt $i $j) -eq 1 ]]; then
#             echo $i $j
#             CUDA_VISIBLE_DEVICES=4 python train.py --img "hyperspectral/thz_new/resol1951_2_e_${i}THz.png" --target "hyperspectral/thz_new/resol1951_2_e_${j}THz.png" --model ZSSR --exp o2o_leaky
#             CUDA_VISIBLE_DEVICES=4 python test.py --ckpt_path checkpoints_main/ZSSR_o2o_leaky/resol1951_2_e_${i}_to_${j}THz_latest.pt --save_dir o2o_latest_ZSSR_leaky --model ZSSR
#         # fi
#     done
# done


echo "Exp1: One-to-One / Center Norm"
function float_lt() {
    awk -v a="$1" -v b="$2" 'BEGIN {print (a < b)}'
}
for i in $(seq 0.1 0.1 3.0); do
    for j in $(seq 0.1 0.1 3.0); do
        # if [[ $(float_lt $i $j) -eq 1 ]]; then
            echo $i $j
            CUDA_VISIBLE_DEVICES=4 python train.py --img "hyperspectral/centernorm_step0.1_reshapev2.0x5_shift1/resol1951_2_e_${i}THz.png" --target "hyperspectral/centernorm_step0.1_reshapev2.0x5_shift1/resol1951_2_e_${j}THz.png" --model ZSSR --exp o2o_centernorm
            CUDA_VISIBLE_DEVICES=4 python test.py --ckpt_path checkpoints_main/ZSSR_o2o_centernorm/resol1951_2_e_${i}_to_${j}THz_latest.pt --save_dir o2o_centernorm --model ZSSR
        # fi
    done
done

# CUDA_VISIBLE_DEVICES=4 python train.py --img "hyperspectral/centernorm_step0.1_reshapev2.0x5_shift1/resol1951_2_e_1.5THz.png" --target "hyperspectral/centernorm_step0.1_reshapev2.0x5_shift1/resol1951_2_e_2.0THz.png" --model ZSSR --exp o2o_centernorm
# CUDA_VISIBLE_DEVICES=4 python test.py --ckpt_path checkpoints_main/ZSSR_o2o_centernorm/resol1951_2_e_1.5_to_2.0THz_latest.pt --save_dir o2o_centernorm --model ZSSR

# CUDA_VISIBLE_DEVICES=0 python train.py --img "denoised/denoised_resol1951_2_e_0.5THz.png" --target "denoised/denoised_resol1951_2_e_1.0THz.png"
# CUDA_VISIBLE_DEVICES=0 python train.py --img "denoised/denoised_resol1951_2_e_1.0THz.png" --target "denoised/denoised_resol1951_2_e_1.5THz.png"
# CUDA_VISIBLE_DEVICES=0 python train.py --img "denoised/denoised_resol1951_2_e_1.5THz.png" --target "denoised/denoised_resol1951_2_e_2.0THz.png"
# CUDA_VISIBLE_DEVICES=0 python train.py --img "denoised/denoised_resol1951_2_e_2.0THz.png" --target "denoised/denoised_resol1951_2_e_2.5THz.png" 
# CUDA_VISIBLE_DEVICES=0 python test.py --ckpt_path checkpoints/ZSSR/denoised_resol1951_2_e_1.5_to_2.0THz.pt --test_img denoised/denoised_resol1951_2_e_2.0THz.png --save_dir o2o

# CUDA_VISIBLE_DEVICES=0 python train.py --img "denoised/sh_resol1951_2_e_1.6THz.png" --target "denoised/sh_resol1951_2_e_2.0THz.png"

# : ' ===== Experiment 2: Band-to-Band ===== '
# echo "============ Exp2: Band-to-Band ============"
# for i in $(seq 0.2 0.1 3.0)
# do
#     i2=$(echo "scale=1; $i + 1.0" | bc -l)
#     i2=$(printf "%.1f" $i2)
#     for j in $(seq 0.2 0.1 3.0) # Short band
#     # for j in $(seq 1.1 0.1 1.2) # Wide band
#     do
#         j2=$(echo "scale=1; $j + 1.0" | bc -l) # 0.1, 0.5, 1.0
#         j2=$(printf "%.1f" $j2)
#         echo $i $i2 $j $j2
#         CUDA_VISIBLE_DEVICES=0 python train_band.py --inp_minmax $i,$i2 --gt_minmax $j,$j2 --num_batches 25000 --exp b2b
#         CUDA_VISIBLE_DEVICES=0 python test.py --ckpt_path checkpoints/ZSSR_b2b/resol1951_2_e_${i},${i2}_to_${j},${j2}.pt --save_dir wb2wb
#     done
# done

# : ' ===== Experiment 3-1: One-to-Band ===== '
# echo "============ Exp3-1: One-to-Short Band ============"
# for i in $(seq 1.4 0.1 3.0); do
#     for j1 in $(seq 0.2 0.1 3.0); do
#         j2=$(echo "scale=1; $j1 + 0.1" | bc -l)
#         j2=$(printf "%.1f" $j2)
#         echo $i $j1 $j2
#         CUDA_VISIBLE_DEVICES=9 python train_band.py --inp_minmax $i,$i --gt_minmax $j1,$j2 --num_batches 25000 --exp o2sb
#         CUDA_VISIBLE_DEVICES=9 python test.py --ckpt_path checkpoints/ZSSR_o2sb/resol1951_2_e_${i},${i}_to_${j1},${j2}.pt --save_dir o2sb
#     done
# done

# echo "============ Exp3-1: One-to-Midium Band ============"
# for i in $(seq 1.5 0.1 3.0); do
#     for j1 in $(seq 0.2 0.1 2.8); do
#         j2=$(echo "scale=1; $j1 + 0.3" | bc -l)
#         j2=$(printf "%.1f" $j2)
#         echo $i $j1 $j2
#         CUDA_VISIBLE_DEVICES=9 python train_band.py --inp_minmax $i,$i --gt_minmax $j1,$j2 --num_batches 25000 --exp o2mb
#         CUDA_VISIBLE_DEVICES=9 python test.py --ckpt_path checkpoints/ZSSR_o2mb/resol1951_2_e_${i},${i}_to_${j1},${j2}.pt --save_dir o2mb
#     done
# done

# echo "============ Exp3-1: One-to-Wide Band ============"
# for i in $(seq 1.6 0.1 3.0); do
#     for j1 in $(seq 0.2 0.1 3.0); do
#         j2=$(echo "scale=1; $j1 + 0.5" | bc -l)
#         j2=$(printf "%.1f" $j2)
#         echo $i $j1 $j2
#         CUDA_VISIBLE_DEVICES=9 python train_band.py --inp_minmax $i,$i --gt_minmax $j1,$j2 --num_batches 25000 --exp o2wb
#         CUDA_VISIBLE_DEVICES=9 python test.py --ckpt_path checkpoints/ZSSR_o2wb/resol1951_2_e_${i},${i}_to_${j1},${j2}.pt --save_dir o2wb
#     done
# done


# : ' ===== Experiment 3-2: Band-to-One ===== '
# echo "============ Exp3-2: Short Band-to-One ============"
# for i1 in $(seq 1.4 0.1 3.0); do
#     i2=$(echo "scale=1; $i1 + 0.1" | bc -l)
#     i2=$(printf "%.1f" $i2)
#     for j in $(seq 0.2 0.1 3.0); do
#         echo $i1 $i2 $j
#         CUDA_VISIBLE_DEVICES=8 python train_band.py --inp_minmax $i1,$i2 --gt_minmax $j,$j --num_batches 25000 --exp sb2o
#         CUDA_VISIBLE_DEVICES=8 python test.py --ckpt_path checkpoints/ZSSR_sb2o/resol1951_2_e_${i1},${i2}_to_${j},${j}.pt --save_dir sb2o
#     done
# done

# echo "============ Exp3-2: Midium Band-to-One ============"
# for i1 in $(seq 1.4 0.1 3.0); do
#     i2=$(echo "scale=1; $i1 + 0.3" | bc -l)
#     i2=$(printf "%.1f" $i2)
#     for j in $(seq 0.2 0.1 3.0); do
#         echo $i1 $i2 $j
#         CUDA_VISIBLE_DEVICES=8 python train_band.py --inp_minmax $i1,$i2 --gt_minmax $j,$j --num_batches 25000 --exp mb2o
#         CUDA_VISIBLE_DEVICES=8 python test.py --ckpt_path checkpoints/ZSSR_mb2o/resol1951_2_e_${i1},${i2}_to_${j},${j}.pt --save_dir mb2o
#     done
# done

# echo "============ Exp3-2: Wide Band-to-One ============"
# for i1 in $(seq 1.4 0.1 3.0); do
#     i2=$(echo "scale=1; $i1 + 0.5" | bc -l)
#     i2=$(printf "%.1f" $i2)
#     for j in $(seq 0.2 0.1 3.0); do
#         echo $i1 $i2 $j
#         CUDA_VISIBLE_DEVICES=8 python train_band.py --inp_minmax $i1,$i2 --gt_minmax $j,$j --num_batches 25000 --exp wb2o
#         CUDA_VISIBLE_DEVICES=8 python test.py --ckpt_path checkpoints/ZSSR_wb2o/resol1951_2_e_${i1},${i2}_to_${j},${j}.pt --save_dir wb2o
#     done
# done


# : ' ===== Experiment 4: Recurrent v1 ===== '
# # start=1.5; freqs=(1.8 2.0 2.2) # 실험1
# # starts=(seq 0.2 0.1 3.0); freqs=(1.6 1.7 1.9 2.3) # 실험2
# for start in $(seq 0.2 0.1 3.0); do
# for freq in $(seq 0.2 0.1 3.0); do
# cp "results_main/o2o_latest_ZSSR/${freq}/resol1951_2_e_${freq}THz(${start}to${freq}).png" results_main/recurrent/
# echo "-------------- Recurrent v1; ${freq} ---------------"
# CUDA_VISIBLE_DEVICES=0 \
# python train.py \
#     --img hyperspectral/thz_new/resol1951_2_e_${freq}THz.png \
#     --target "results_main/o2o_latest_ZSSR/${freq}/resol1951_2_e_${freq}THz(${start}to${freq}).png" \
#     --exp recurrent
# CUDA_VISIBLE_DEVICES=0 \
# python test.py \
#     --test_img "results_main/o2o_latest_ZSSR/${freq}/resol1951_2_e_${freq}THz(${start}to${freq}).png" \
#     --ckpt_path "checkpoints_main/ZSSR_recurrent/resol1951_2_e_${freq}THzto${freq}THz(${start}to${freq})_latest.pt" \
#     --save_dir recurrent
# echo "-------------- Re-recurrent v1; ${freq} ---------------"
# CUDA_VISIBLE_DEVICES=0 \
# python train.py \
#     --img "results_main/o2o_latest_ZSSR/${freq}/resol1951_2_e_${freq}THz(${start}to${freq}).png" \
#     --target "results_main/recurrent/resol1951_2_e_${freq}THz(${start}to${freq})(${freq}THzto${freq}THz(${start}to${freq})).png" \
#     --exp recurrent
# python test.py \
#     --test_img "results_main/recurrent/resol1951_2_e_${freq}THz(${start}to${freq})(${freq}THzto${freq}THz(${start}to${freq})).png" \
#     --ckpt_path "checkpoints_main/ZSSR_recurrent/resol1951_2_e_${freq}THz(${start}to${freq})to${freq}THz(${start}to${freq})(${freq}THzto${freq}THz(${start}to${freq}))_latest.pt" \
#     --save_dir recurrent
# done; done

# : ' ===== Experiment 4: Recurrent v2 ===== '
# # freqs=(1.5 1.7 1.8 1.9) # 1.5 추가
# # for freq in ${freqs[@]}; do
# # cp "results_main/o2o_latest_ZSSR/2.0/resol1951_2_e_2.0THz(${freq}to2.0).png" results_main/recurrentv2/
# # echo "-------------- Recurrent v2; ${freq} ---------------"
# # CUDA_VISIBLE_DEVICES=0 \
# # python train.py --img hyperspectral/thz_new/resol1951_2_e_${freq}THz.png --target "results_main/o2o_latest_ZSSR/2.0/resol1951_2_e_2.0THz(${freq}to2.0).png" --exp recurrentv2
# # CUDA_VISIBLE_DEVICES=0 \
# # python test.py \
# #     --test_img "results_main/o2o_latest_ZSSR/2.0/resol1951_2_e_2.0THz(${freq}to2.0).png" \
# #     --ckpt_path "checkpoints_main/ZSSR_recurrentv2/resol1951_2_e_${freq}THzto2.0THz(${freq}to2.0)_latest.pt" \
# #     --save_dir recurrentv2
# # echo "-------------- Re-recurrent v2; ${freq} ---------------"
# # CUDA_VISIBLE_DEVICES=0 \
# # python train.py --img hyperspectral/thz_new/resol1951_2_e_${freq}THz.png --target "results_main/recurrentv2/resol1951_2_e_2.0THz(${freq}to2.0)(${freq}THzto2.0THz(${freq}to2.0)).png" --exp recurrentv2
# # python test.py \
# #     --test_img "results_main/recurrentv2/resol1951_2_e_2.0THz(${freq}to2.0)(${freq}THzto2.0THz(${freq}to2.0)).png" \
# #     --ckpt_path "checkpoints_main/ZSSR_recurrentv2/resol1951_2_e_${freq}THzto2.0THz(${freq}to2.0)(${freq}THzto2.0THz(${freq}to2.0))_latest.pt" \
# #     --save_dir recurrentv2
# # done


# : ' ===== Experiment 5: Volt data ===== '
# # CUDA_VISIBLE_DEVICES=0 python train.py --img thz/resol_10e_1.0THz.png --target thz/resol_100e_1.4THz.png --num_batches 1000


# : ' ===== Two target ===== '
# # python train_two_target.py --img "thz/resol1951_2_e_2.3THz.png" --low_target "thz/resol1951_2_e_2.0THz.png" --high_target "thz/resol1951_2_e_2.8THz.png" --exp two_target --lamb 0.5
# # python test.py --ckpt_path checkpoints/ZSSR_two_target/resol1951_2_e_2.3_to_2.0and2.8THz_latest.pt --test_img thz/resol1951_2_e_2.3THz.png --save_dir two_target


# : ' ===== Experiment 6: One-to-Denoised ===== '
# CUDA_VISIBLE_DEVICES=9 python train.py --img "hyperspectral/thz_new/resol1951_2_e_2.0THz.png" --target "denoised/denoised_signal_jsh/2.5050THz.png" --model ZSSR --exp o2d &&
# CUDA_VISIBLE_DEVICES=9 python test.py --ckpt_path checkpoints_main/ZSSR_o2d/resol1951_2_e_2.0_to_2.5050THz_latest.pt --save_dir o2d
# CUDA_VISIBLE_DEVICES=9 python test.py --ckpt_path checkpoints_main/ZSSR_o2d/resol1951_2_e_2.0_to_2.5050THz_latest.pt --save_dir o2d --denoised

# : ' ===== Experiment 6: Denoised-to-Denoised ===== '
# CUDA_VISIBLE_DEVICES=9 python train.py --img "denoised/denoised_signal_jsh/2.0040THz.png" --target "denoised/denoised_signal_jsh/2.5050THz.png" --model ZSSR --exp d2d &&
# CUDA_VISIBLE_DEVICES=9 python test.py --ckpt_path checkpoints_main/ZSSR_d2d/_2.0040_to_2.5050THz_latest.pt --save_dir d2d --denoised

# CUDA_VISIBLE_DEVICES=9 python train.py --img "denoised/denoised_signal_jsh/2.0040THz.png" --target "denoised/denoised_signal_jsh/3.0060THz.png" --model ZSSR --exp d2d &&
# CUDA_VISIBLE_DEVICES=9 python test.py --ckpt_path checkpoints_main/ZSSR_d2d/_2.0040_to_3.0060THz_latest.pt --save_dir d2d --denoised

# CUDA_VISIBLE_DEVICES=9 python train.py --img "denoised/denoised_signal_jsh/2.0040THz.png" --target "denoised/denoised_signal_jsh/3.0060THz.png" --model ZSSR --exp d2d &&
# CUDA_VISIBLE_DEVICES=9 python test.py --ckpt_path checkpoints_main/ZSSR_d2d/_2.0040_to_3.0060THz_latest.pt --save_dir d2d --denoised

