: ' ===== Experiment 1: One-to-One ===== '
# echo "Exp1: One-to-One"
# for i in $(seq 0.2 0.1 0.3)
# do
#     for j in $(seq 0.2 0.1 3.0)
#     do
#         echo $i $j
#         CUDA_VISIBLE_DEVICES=0 python train.py --img thz/resol1951_2_e_${i}THz.png --target thz/resol1951_2_e_${j}THz.png
#     done
# done

: ' ===== Experiment 2: Band-to-Band ===== '
# echo "============ Exp2: Band-to-Band ============"
# for i in $(seq 0.3 0.1 0.5)
# do
#     i2=$(echo "scale=1; $i + 0.1" | bc -l)
#     i2=$(printf "%.1f" $i2)
#     for j in $(seq 1.1 0.1 2.2) # Short band
#     # for j in $(seq 1.1 0.1 1.2) # Wide band
#     do
#         j2=$(echo "scale=1; $j + 0.1" | bc -l) # 1.0
#         j2=$(printf "%.1f" $j2)
#         echo $i $i2 $j $j2
#         CUDA_VISIBLE_DEVICES=0 python train_band.py --inp_minmax $i,$i2 --gt_minmax $j,$j2 --num_batches 25000
#     done
# done

: ' ===== Experiment 3: Band-to-One ===== '
# echo "============ Exp3: Band-to-One ============"
# for i in $(seq 0.2 0.1 0.5); do
#     i2=$(echo "scale=1; $i + 0.1" | bc -l)
#     i2=$(printf "%.1f" $i2)
#     for j in $(seq 0.1 0.1 2.4); do
#         i3=$(echo "scale=1; $j + $i2" | bc -l)
#         i3=$(printf "%.1f" $i3)
#         echo $i $i2 $j $j2
#         CUDA_VISIBLE_DEVICES=0 python train_band.py --inp_minmax $i,$i2 --gt_minmax $i3,$i3 --num_batches 20000
#     done
# done

: ' ===== Experiment 4: Recurrent ===== '
# CUDA_VISIBLE_DEVICES=0 python train.py --img thz/resol1951_2_e_0.2THz.png --target thz/resol1951_2_e_1.0THz.png
# CUDA_VISIBLE_DEVICES=0 python train.py --img thz/resol1951_2_e_0.3THz.png --target thz/resol1951_2_e_2.1THz.png

# CUDA_VISIBLE_DEVICES=0 python train.py --img thz/resol1951_2_e_2.0THz.png --target "results/exp4_dbdn/resol1951_2_e_2.0THz(1.5to2.0)_zssr.png"
# CUDA_VISIBLE_DEVICES=0 python train.py --img thz/resol1951_2_e_1.5THz.png --target "results/exp4_dbdn/resol1951_2_e_2.0THz(1.5to2.0)_zssr.png"

# re-recurrent
# CUDA_VISIBLE_DEVICES=0 python train.py --img "results/exp4_dbdn/resol1951_2_e_2.0THz(1.5to2.0)_zssr.png" --target "results/exp4_dbdn/resol1951_2_e_2.0THz(1.5to2.0)_zssr(2.0tozssr)_zssr.png"


: ' ===== Experiment 5: New data ===== '
CUDA_VISIBLE_DEVICES=0 python train.py --img thz/resol_10e_1.0THz.png --target thz/resol_100e_1.4THz.png --num_batches 1000