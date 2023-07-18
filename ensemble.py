import argparse
import numpy as np
import os
from PIL import Image

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img1', type=str, help='Path to img1')
    parser.add_argument('--img2', type=str, help='Path to img2')
    parser.add_argument('--weight', type=float, help='Weight for result ensemble')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    
    path1 = args.img1
    path2 = args.img2
    
    img1 = Image.open(path1)
    img2 = Image.open(path2)
    
    img1_np = np.array(img1)
    img2_np = np.array(img2)

    w = args.weight
    
    result_np = w * img1_np + (1 - w) * img2_np
    result_img = Image.fromarray(result_np.astype(np.uint8))
        
    basename = os.path.basename(path1).split('(')
    img_name = basename[0]
    low_freq = path1.split(img_name)[1].split('.png')[0]
    high_freq = path2.split(img_name)[1].split('.png')[0]

    save_name = f'{img_name}[{w}x{low_freq}+{round(1-w, 1)}x{high_freq}].png'
    os.makedirs('results/weighted_sum', exist_ok=True)
    result_img.save('results/weighted_sum/'+save_name)