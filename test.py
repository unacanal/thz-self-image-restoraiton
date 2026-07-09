import os
import numpy as np
from net import ZSSRNet, CAIR, ZSN2N
import torch
from torch.autograd import Variable
import PIL
from torchvision import transforms
import argparse

try:
    BICUBIC = PIL.Image.Resampling.BICUBIC
except AttributeError:  # Pillow < 9.1
    BICUBIC = PIL.Image.BICUBIC


def test(model, img, save_dir, save_name, scale=1):
    model.eval()

    if scale > 1:
        # ZSSR inference: bicubic-upscale the input by `scale`, then restore the
        # high-frequency details at the enlarged resolution.
        w, h = img.size
        img = img.resize((w * scale, h * scale), BICUBIC)

    img = transforms.ToTensor()(img)
    img = torch.unsqueeze(img, 0)
    input = Variable(img.cuda())
    print(input.shape)
    residual = model(input)
    output = input + residual

    output = output.cpu().data[0, :, :, :]
    o = output.numpy()
    o[np.where(o < 0)] = 0.0
    o[np.where(o > 1)] = 1.0
    output = torch.from_numpy(o)
    output = transforms.ToPILImage()(output)
    
    if args.volt:
        prefix = 'results_volt'
    else:
        prefix = 'results_main'
        
    os.makedirs(f'{prefix}/{save_dir}', exist_ok=True)
    output.save(f'{prefix}/{save_dir}/{save_name}.png')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_img', type=str, help='Path to test img', default=None, required=False)
    parser.add_argument('--ckpt_path', type=str, help='Path to checkpoint')
    parser.add_argument('--save_dir', type=str, help='Dir to save result')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--volt', type=str, default=None, help='Voltage')
    parser.add_argument('--denoised', action='store_true', help='Denoised')
    parser.add_argument('--scale', type=int, default=1, \
        help='ZSSR super-resolution scale factor (1 = no upscaling)')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
        
    if args.model == 'CAIR':
            model = CAIR(img_channel=1, width=16, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 1], dec_blk_nums=[1, 1, 1, 1])
    elif args.model == 'ZSN2N':
        model = ZSN2N(n_chan=1)
    else:
        model = ZSSRNet(input_channels=1)
        
    model.cuda()

    model.load_state_dict(torch.load(args.ckpt_path))
    if not args.test_img:
        # if args.test_img not exists
        testset = np.arange(0.1, 1.5, 0.1)
        testset2 = np.arange(1.4, 3.1, 0.1)
        testset = np.append(testset, testset2)

        if args.denoised:
            testset = ['2.0040', '2.5050']

        for k in testset:
            
            if args.volt:
                k = round(k, 2)
                test_img_path = f'hyperspectral/{args.volt}_30ps_e/{args.volt}_30ps_e_{k}THz.png'
            elif args.denoised:
                test_img_path = f'denoised/denoised_signal_jsh/{k}THz.png'
            else:
                k = round(k, 2)
                test_img_path =  f'hyperspectral/centernorm_step0.1_reshapev2.0x5_shift1/resol1951_2_e_{k}THz.png'
                

            test_img = PIL.Image.open(test_img_path)

            test_img_name = os.path.splitext(os.path.basename(test_img_path))[0]
            save_name = test_img_name + \
                '('+''.join(os.path.splitext(os.path.basename(args.ckpt_path))[0].split('_')[-4:]).split('THz')[0]+')'
            
            save_dir = os.path.join(args.save_dir, str(k))
            test(model, test_img, save_dir, save_name, scale=args.scale)

    else:
        test_img = PIL.Image.open(args.test_img)

        test_img_name = os.path.splitext(os.path.basename(args.test_img))[0]
        
        if 'recurrent' in args.save_dir:
            save_name = test_img_name + \
                '('+ os.path.basename(args.ckpt_path).split('_')[-2] + ')'
            print(test_img_name, args.ckpt_path)
            print(save_name)
        else:
            save_name = test_img_name + \
                '('+ os.path.basename(args.ckpt_path).split('THz')[0] + ')'
        
        if os.path.basename(args.ckpt_path).split('_')[-2][:4] == 'lamb':
            save_name += os.path.basename(args.ckpt_path).split('_')[-2]

        test(model, test_img, args.save_dir, save_name, scale=args.scale)