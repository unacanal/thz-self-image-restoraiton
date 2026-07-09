# CUDA_VISIBLE_DEVICES=0 python test.py --test_img thz/resol1951_2_e_1.0THz.png --ckpt_path checkpoints/resol19512e_0.3_to_1.0THz.pt
# CUDA_VISIBLE_DEVICES=0 python test.py --test_img thz/resol1951_2_e_1.0THz.png --ckpt_path checkpoints/resol19512e_0.4_to_1.0THz.pt
# CUDA_VISIBLE_DEVICES=0 python test.py --test_img thz/resol1951_2_e_1.0THz.png --ckpt_path checkpoints/resol19512e_0.3_to_2.1THz.pt
# CUDA_VISIBLE_DEVICES=0 python test.py --test_img thz/resol1951_2_e_1.0THz.png --ckpt_path checkpoints/resol19512e_0.4_to_2.1THz.pt

# CUDA_VISIBLE_DEVICES=0 python test.py --test_img thz/resol1951_2_e_2.1THz.png --ckpt_path checkpoints/resol19512e_0.3_to_1.0THz.pt
# CUDA_VISIBLE_DEVICES=0 python test.py --test_img thz/resol1951_2_e_2.1THz.png --ckpt_path checkpoints/resol19512e_0.4_to_1.0THz.pt
# CUDA_VISIBLE_DEVICES=0 python test.py --test_img thz/resol1951_2_e_2.1THz.png --ckpt_path checkpoints/resol19512e_0.3_to_2.1THz.pt
# CUDA_VISIBLE_DEVICES=0 python test.py --test_img thz/resol1951_2_e_2.1THz.png --ckpt_path checkpoints/resol19512e_0.4_to_2.1THz.pt

: ' ===== Experiment 1: One-to-One ===== '
echo "Experiment 1: One-to-One"
function float_lt() {
    awk -v a="$1" -v b="$2" 'BEGIN {print (a < b)}'
}

for i in $(seq 0.1 0.1 3.0); do
    for j in $(seq 0.1 0.1 3.0); do
        # if [[ $(float_lt $i $j) -eq 1 ]]; then
            echo $i $j
            CUDA_VISIBLE_DEVICES=2 python test.py --ckpt_path checkpoints/ZSSR_o2o/resol1951_2_e_${i}_to_${j}THz_best.pt --save_dir o2o_best; CUDA_VISIBLE_DEVICES=2 python test.py --ckpt_path checkpoints/ZSSR_o2o/resol1951_2_e_${i}_to_${j}THz_latest.pt --save_dir o2o_latest
        # fi
    done
done

# CUDA_VISIBLE_DEVICES=0 python test.py --ckpt_path checkpoints/ZSSR/denoised_resol1951_2_e_1.5_to_2.0THz.pt --test_img denoised/denoised_resol1951_2_e_2.0THz.png --save_dir o2o

: ' Test with CW image '
# Knife 300GHz
for i in $(seq 0.1 0.1 3.0); do
    for j in $(seq 0.1 0.1 3.0); do
        echo $i $j
        CUDA_VISIBLE_DEVICES=0 python test.py --test_img hyperspectral/knife/0.31THz.png --ckpt_path checkpoints_main/ZSSR_o2o/resol1951_2_e_${i}_to_${j}THz_latest.pt --save_dir CW
    done
done



: ' ===== Experiment 2: Band-to-Band ===== '
# echo "Experiment 2: Band-to-Band"
# for i in $(seq 0.3 0.1 0.5); do
#     i2=$(echo "scale=1; $i + 0.1" | bc -l)
#     i2=$(printf "%.1f" $i2)
#     # echo "--->SB to SB"
#     # for j in $(seq 1.1 0.1 2.2); do # Short Band
#     #     j2=$(echo "scale=1; $j + 0.1" | bc -l)
#     #     j2=$(printf "%.1f" $j2)
#     #     echo $i $i2 $j $j2
#     #     # for k in $(seq 0.2 0.2 1.0) $(seq 1.2 0.2 2.2); do
#     #     CUDA_VISIBLE_DEVICES=0 python test.py --ckpt_path checkpoints/resol1951_2_e_${i},${i2}_to_${j},${j2}.pt --save_dir exp2_sb2sb
#     #     # done
#     # done
#     echo "--->SB to WB"
#     for j in $(seq 1.1 0.1 1.2); do # Wide band
#         j2=$(echo "scale=1; $j + 1.0" | bc -l)
#         j2=$(printf "%.1f" $j2)
#         echo $i $i2 $j $j2
#         CUDA_VISIBLE_DEVICES=0 python test.py --ckpt_path checkpoints/resol1951_2_e_${i},${i2}_to_${j},${j2}.pt --save_dir exp2_sb2wb
#     done
# done


: ' ===== Experiment 3: Band-to-One ===== '
# echo "Experiment 3: Band-to-One"
# for i in $(seq 0.2 0.1 0.5); do
#     i2=$(echo "scale=1; $i + 0.1" | bc -l)
#     i2=$(printf "%.1f" $i2)
#     for j in $(seq 0.6 0.1 2.4); do # originally from 0.1
#         i3=$(echo "scale=1; $j + $i2" | bc -l)
#         i3=$(printf "%.1f" $i3)
#         echo $i $i2 $i3

#         CUDA_VISIBLE_DEVICES=0 python test.py --ckpt_path checkpoints/resol1951_2_e_${i},${i2}_to_${i3},${i3}.pt --save_dir exp3_b2o
#     done
# done


: ' ===== Experiment 4: Deblur+Denoise ===== '
# echo "Experiment 4: Deblur+Denoise"

#1 0.4->1.0
#2 2.0->1.0
#3 0.4->2.0

# Method1: apply#1 (Deblurring)
# Method2: apply#2 then apply#1 (Denoising)
# Method3: apply#3 then apply#2 (Deblurring)

# How will the Method2 result in?
# Will the Method1 and the Method3 result in the same output?

# Method1
# CUDA_VISIBLE_DEVICES=0 python test.py --test_img thz/resol1951_2_e_0.8THz.png --ckpt_path checkpoints/resol19512e_0.4_to_1.0THz.pt --save_dir exp4_dbdn
# CUDA_VISIBLE_DEVICES=0 python test.py --test_img 'results/exp4_dbdn/resol1951_2_e_0.8THz(0.4to1.0)_zssr.png' --ckpt_path checkpoints/resol19512e_1.0_to_2.4THz.pt --save_dir exp4_dbdn
# Method2
# CUDA_VISIBLE_DEVICES=0 python test.py --test_img thz/resol1951_2_e_1.8THz.png --ckpt_path checkpoints/resol19512e_2.4_to_1.0THz.pt --save_dir exp4_dbdn
# CUDA_VISIBLE_DEVICES=0 python test.py --test_img 'results/exp4_dbdn/resol1951_2_e_1.8THz(2.0to1.0)_zssr.png' --ckpt_path checkpoints/resol19512e_0.4_to_1.0THz.pt --save_dir exp4_dbdn
# Method3
# CUDA_VISIBLE_DEVICES=0 python test.py --test_img thz/resol1951_2_e_0.8THz.png --ckpt_path checkpoints/resol19512e_1.0_to_2.0THz.pt --save_dir exp4_dbdn
# CUDA_VISIBLE_DEVICES=0 python test.py --test_img 'results/exp4_dbdn/resol1951_2_e_0.8THz(1.0to2.0)_zssr.png' --ckpt_path checkpoints/resol19512e_2.0_to_1.0THz.pt --save_dir exp4_dbdn


: ' ===== Experiment 4: Recurrent ===== '
CUDA_VISIBLE_DEVICES=0 \
python test.py \
    --test_img thz/resol1951_2_e_2.0THz.png \
    --ckpt_path checkpoints_main/ZSSR_o2o/resol1951_2_e_1.5_to_2.0THz_latest.pt \
    --save_dir recurrent
# CUDA_VISIBLE_DEVICES=0 python test.py --test_img 'results/exp4_dbdn/resol1951_2_e_1.4THz(0.7to1.4)_zssr.png' --ckpt_path checkpoints/resol19512e_1.4_to_2.0THz.pt --save_dir exp4_dbdn

### Recurrent 1 ###
# i=2.0
# j1=1.5
# j2=2.0
# j3=2.5
# k=2.4
# CUDA_VISIBLE_DEVICES=0 python test.py --test_img thz/resol1951_2_e_${i}THz.png --ckpt_path checkpoints/resol19512e_${j1}_to_${j2}THz.pt --save_dir exp4_dbdn
# CUDA_VISIBLE_DEVICES=0 python test.py --test_img "results/exp4_dbdn/resol1951_2_e_${i}THz(${j1}to${j2})_zssr.png" --ckpt_path checkpoints/resol19512e_${j2}_to_${j3}THz.pt --save_dir exp4_dbdn
# CUDA_VISIBLE_DEVICES=0 python test.py --test_img "results/exp4_dbdn/resol1951_2_e_${i}THz(${j1}to${j2})_zssr(${j2}to${j3})_zssr.png" --ckpt_path checkpoints/resol19512e_${j3}_to_${k}THz.pt --save_dir exp4_dbdn

### Recurrent 2 ###
# CUDA_VISIBLE_DEVICES=0 python test.py --test_img "results/exp4_dbdn/resol1951_2_e_2.0THz(1.5to2.0)_zssr.png"  --ckpt_path checkpoints/resol1951_2_e_1.5_to_zssr.pt --save_dir exp4_dbdn

# CUDA_VISIBLE_DEVICES=0 python test.py --test_img "results/exp4_dbdn/resol1951_2_e_2.0THz(1.5to2.0)_zssr.png"  --ckpt_path checkpoints/resol1951_2_e_2.0_to_zssr.pt --save_dir exp4_dbdn
# re-recurrent
# CUDA_VISIBLE_DEVICES=0 python test.py --test_img "results/exp4_dbdn/resol1951_2_e_2.0THz(1.5to2.0)_zssr(2.0tozssr)_zssr.png"  --ckpt_path "checkpoints/resol1951_2_e_2.0THz(1.5to2.0)_zssr_to_zssr.pt" --save_dir exp4_dbdn

: ' ===== Experiment 5: New data ===== '
# python test.py --ckpt_path checkpoints/resol19512e_0.5_to_1.0THz.pt --test_img thz/resol_1e_0.6THz.png --save_dir exp5
# python test.py --ckpt_path checkpoints/resol19512e_0.4_to_0.8THz.pt \
#     --test_img thz/resol_1e_0.6THz.png --save_dir exp5
# python test.py --ckpt_path checkpoints/resol_10e_1.0_to_resol_100e_1.0THz.pt --test_img thz/resol_100e_1.0THz.png --save_dir exp5
# python test.py --ckpt_path checkpoints/resol_10e_1.0_to_resol_100e_1.0THz.pt --test_img thz/resol_10e_1.0THz.png --save_dir exp5
# python test.py --ckpt_path checkpoints/ZSSR/resol1951_2_e_1.5_to_2.0THz_15000.pt --test_img thz/resol1951_2_e_2.0THz.png --save_dir exp6

: ' ==== Other data inference ==== '
# Bottle
freq=2.0
for i in $(seq 0.1 0.1 3.0); do
    for j in $(seq 0.1 0.1 3.0); do
        echo $i $j
        CUDA_VISIBLE_DEVICES=0 python test.py \
        --test_img hyperspectral/bottle/3bottle-e_${freq}THz.png \
        --ckpt_path checkpoints_main/ZSSR_o2o/resol1951_2_e_${i}_to_${j}THz_latest.pt \
        --save_dir "CW/bottle/${freq}"
    done
done

# legbird
for i in $(seq 0.1 0.1 3.0); do
    for j in $(seq 0.1 0.1 3.0); do
        echo $i $j
        CUDA_VISIBLE_DEVICES=0 python test.py \
        --test_img hyperspectral/legbird/3legbird-e_0.5THz.png \
        --ckpt_path checkpoints_main/ZSSR_o2o/resol1951_2_e_${i}_to_${j}THz_latest.pt \
        --save_dir CW/legbird/0.5
    done
done