#!/bin/bash
result_dir='/root/Projects/thz-self-image-restoration/results_main/o2o_latest_ZSSR'
# result_dir='/root/Projects/thz-self-image-restoration/results_main/o2o_centernorm'
python BRISQUE/Python/libsvm/python/all_iqa.py "${result_dir}/0.5" > o2o_0.5THz.log
python BRISQUE/Python/libsvm/python/all_iqa.py "${result_dir}/1.0" > o2o_1.0THz.log
python BRISQUE/Python/libsvm/python/all_iqa.py "${result_dir}/1.5" > o2o_1.5THz.log
python BRISQUE/Python/libsvm/python/all_iqa.py "${result_dir}/2.0" > o2o_2.0THz.log
# python BRISQUE/Python/libsvm/python/all_iqa.py "${result_dir}/2.0" > o2o_centernorm_2.0THz.log
python BRISQUE/Python/libsvm/python/all_iqa.py "${result_dir}/2.2" > o2o_2.2THz.log
python BRISQUE/Python/libsvm/python/all_iqa.py "${result_dir}/2.3" > o2o_2.3THz.log
python BRISQUE/Python/libsvm/python/all_iqa.py "${result_dir}/2.5" > o2o_2.5THz.log
python BRISQUE/Python/libsvm/python/all_iqa.py "${result_dir}/3.0" > o2o_3.0THz.log

# Bottle
# python BRISQUE/Python/libsvm/python/all_iqa.py "/root/why/thz-self-image-restoration/results_main/CW/bottle/1.1" > bottle_1.1THz.log

# 
python BRISQUE/Python/libsvm/python/all_iqa.py /root/Projects/thz-self-image-restoration/hyperspectral/bottle > bottle_original.log
python BRISQUE/Python/libsvm/python/all_iqa.py temp > temp/temp.log

python BRISQUE/Python/libsvm/python/all_iqa.py "/root/Projects/thz-self-image-restoration/hyperspectral/thz_new" > original.log


python BRISQUE/Python/libsvm/python/all_iqa.py "/root/Projects/thz-self-image-restoration/hyperspectral/centernorm_step0.1_reshapev2.0x5_shift1" > original.log