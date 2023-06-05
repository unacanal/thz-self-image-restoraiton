import os
import numpy as np
from net import ZSSRNet
from data import PairedDataSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import init
from torchvision import transforms
import tqdm
import argparse


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def adjust_learning_rate(optimizer, new_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def train(model, name, inp_minmax, gt_minmax, num_batches, learning_rate, crop_size):
    loss = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    sampler = PairedDataSampler(name, inp_minmax, gt_minmax, crop_size)

    model.cuda()
    with tqdm.tqdm(total=num_batches, miniters=1, mininterval=0) as progress:
        for iter, (hr, lr) in enumerate(sampler.generate_data()):
            model.zero_grad()

            lr = Variable(lr).cuda()
            hr = Variable(hr).cuda()

            output = model(lr) + lr
            error = loss(output, hr)

            cpu_loss = error.data.cpu().numpy().item()

            progress.set_description("Iteration: {iter} Loss: {loss}, Learning Rate: {lr}".format( \
                iter=iter, loss=cpu_loss, lr=learning_rate))
            progress.update()

            if iter > 0 and iter % 10000 == 0:
                learning_rate = learning_rate / 10
                adjust_learning_rate(optimizer, new_lr=learning_rate)
                print("Learning rate reduced to {lr}".format(lr=learning_rate) )

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
    parser.add_argument('--inp_minmax', type=str, help='Input THz range (tuple)')
    parser.add_argument('--gt_minmax', type=str, help='Target THz range (tuple)')
    parser.add_argument('--name', type=str, default='resol1951_2_e', help='Image name')

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

    model = ZSSRNet(input_channels = 1)
    
    # Weight initialization
    model.apply(weights_init_kaiming)
    

    inp_minmax = tuple(map(float, args.inp_minmax.split(',')))
    gt_minmax = tuple(map(float, args.gt_minmax.split(',')))
    
    save_name = args.name + '_' + args.inp_minmax + '_to_' + args.gt_minmax
    
    train(model, args.name, inp_minmax, gt_minmax, args.num_batches, args.lr, args.crop)
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/'+save_name+'.pt')
    