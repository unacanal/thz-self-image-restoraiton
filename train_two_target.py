import os
import numpy as np
from net import ZSSRNet, CAIR
from data import TripleDataSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import init
import PIL
import sys
from torchvision import transforms
import tqdm
import wandb
import argparse
import warnings
warnings.filterwarnings("ignore")


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def adjust_learning_rate(optimizer, new_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def train(model, img, low_target, high_target, args, save_name):
    num_batches = args.num_batches
    learning_rate = args.lr
    crop_size = args.crop

    loss1 = nn.L1Loss()
    loss2 = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    sampler = TripleDataSampler(img, low_target, high_target, crop_size)
    
    best_loss = float('inf')
    model.cuda()
    with tqdm.tqdm(total=num_batches, miniters=1, mininterval=0) as progress:
        for iter, (gt1, gt2, inp) in enumerate(sampler.generate_data()):
            model.zero_grad()

            inp = Variable(inp).cuda()
            gt1 = Variable(gt1).cuda()
            gt2 = Variable(gt2).cuda()

            output = model(inp) + inp
            error1 = loss1(output, gt1)
            error2 = torch.sqrt(loss2(output, gt2))

            error = (1 - args.lamb) * error1 + args.lamb * error2
            cpu_loss = error.data.cpu().numpy().item()

            progress.set_description("Iteration: {iter} Loss: {loss}, Learning Rate: {lr}".format( \
                iter=iter, loss=cpu_loss, lr=learning_rate))
            progress.update()

            if iter > 0 and iter % 10000 == 0:
                learning_rate = learning_rate / 10
                adjust_learning_rate(optimizer, new_lr=learning_rate)
                print("Learning rate reduced to {lr}".format(lr=learning_rate) )

            if iter % 100 == 0:
                wandb.log({"Train loss": cpu_loss})
            
            if cpu_loss < best_loss:
                best_loss = cpu_loss
                torch.save(model.state_dict(), os.path.join('checkpoints', f'{args.model}_{args.exp}', f'{save_name}_best.pt'))

            error.backward()
            optimizer.step()

            if iter > num_batches:
                print(f"Iteration: {iter}, Loss: {cpu_loss}, Learning Rate: {learning_rate}")
                print('Done training.')
                break
            
            
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_batches', type=int, default=15000, \
        help='Number of batches to run')
    parser.add_argument('--crop', type=int, default=96, \
        help='Random crop size')
    parser.add_argument('--lr', type=float, default=0.00001, \
        help='Base learning rate for Adam')
    parser.add_argument('--img', type=str, help='Path to input img')
    parser.add_argument('--low_target', type=str, help='Path to low target img')
    parser.add_argument('--high_target', type=str, help='Path to high target img')
    parser.add_argument('--test_img', type=str, help='Path to test img')
    parser.add_argument('--model', type=str, default='ZSSR', help='Model name')
    parser.add_argument('--exp', type=str, default='two_target', help='Experiment name')
    parser.add_argument('--lamb', type=float, default=1, help='lambda for L2 loss')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    random_seed = 36
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

    args = get_args()
    
    img = PIL.Image.open(args.img)

    low_target = PIL.Image.open(args.low_target)
    high_target = PIL.Image.open(args.high_target)
    
    num_channels = len(np.array(low_target).shape)
    print('num_channels:', num_channels)
    if num_channels == 3:
        if args.model == 'CAIR':
            model = CAIR(img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 1], dec_blk_nums=[1, 1, 1, 1])
        else:
            model = ZSSRNet(input_channels = 3)
    elif num_channels == 2:
        if args.model == 'CAIR':
            model = CAIR(img_channel=1, width=16, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 1], dec_blk_nums=[1, 1, 1, 1])
        else:
            model = ZSSRNet(input_channels = 1)
    else:
        print("Expecting RGB or gray image, instead got", low_target.size)
        sys.exit(1)

    # Weight initialization
    model.apply(weights_init_kaiming)

    img_name = os.path.splitext(os.path.basename(args.img))[0]
    low_target_name = os.path.splitext(os.path.basename(args.low_target))[0]
    high_target_name = os.path.splitext(os.path.basename(args.high_target))[0]
    
    print(img_name, '=>', low_target_name, high_target_name)

    
    save_name = '_'.join(img_name.split('_')[:-1]) + '_' + \
                    img_name.split('_')[-1].split('THz')[0] + \
                    '_to_' + low_target_name.split('_')[-1].split('THz')[0] + \
                    'and' + high_target_name.split('_')[-1].split('THz')[0] + 'THz_lamb' + str(args.lamb)
    
    print(save_name)
    wandb.init(project='thz-self-image-restoration', name=f'{args.exp}_{save_name}')
    train(model, img, low_target, high_target, args, save_name)
    os.makedirs(f'checkpoints/{args.model}_{args.exp}', exist_ok=True)
    torch.save(model.state_dict(), os.path.join('checkpoints', f'{args.model}_{args.exp}', f'{save_name}_latest.pt'))
    