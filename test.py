import os
import numpy as np
from net import ZSSRNet, CAIR
import torch
from torch.autograd import Variable
import PIL
from torchvision import transforms
import argparse


def test(model, img, save_dir, save_name):
    model.eval()

    img = transforms.ToTensor()(img)
    img = torch.unsqueeze(img, 0)
    input = Variable(img.cuda())
    residual = model(input)
    output = input + residual

    output = output.cpu().data[0, :, :, :]
    o = output.numpy()
    o[np.where(o < 0)] = 0.0
    o[np.where(o > 1)] = 1.0
    output = torch.from_numpy(o)
    output = transforms.ToPILImage()(output)
    output.save(f'results/{save_dir}/{save_name}.png')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_img', type=str, help='Path to test img', default=None, required=False)
    parser.add_argument('--ckpt_path', type=str, help='Path to checkpoint')
    parser.add_argument('--save_dir', type=str, help='Dir to save result')
    parser.add_argument('--model', type=str, help='Model name')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    
    if args.model == 'CAIR':
        model = CAIR(
                    img_channel=1, width=16, middle_blk_num=1,
                    enc_blk_nums=[1, 1, 1, 1], dec_blk_nums=[1, 1, 1, 1]
                ).cuda()
    else:
        model = ZSSRNet(input_channels=1).cuda()

    model.load_state_dict(torch.load(args.ckpt_path))
    if not args.test_img:
        # if args.test_img not exists
        testset = np.arange(0.2, 1.2, 0.2)
        testset2 = np.arange(1.2, 2.2, 0.2)
        testset = np.append(testset, testset2)

        for k in testset:
            k = round(k, 2)
            test_img_path =  f'thz/resol1951_2_e_{k}THz.png'
            print(test_img_path)

            test_img = PIL.Image.open(test_img_path)

            test_img_name = os.path.splitext(os.path.basename(test_img_path))[0]
            save_name = test_img_name + \
                '('+''.join(os.path.splitext(os.path.basename(args.ckpt_path))[0].split('_')[-3:]).split('THz')[0]+')'
                
            test(model, test_img, args.save_dir, save_name)

    else:
        test_img = PIL.Image.open(args.test_img)

        test_img_name = os.path.splitext(os.path.basename(args.test_img))[0]
        save_name = test_img_name + \
            '('+ os.path.basename(args.ckpt_path).split('THz')[0] + ')'
            
        test(model, test_img, args.save_dir, save_name)