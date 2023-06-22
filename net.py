import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class ZSSRNet(nn.Module):
	def __init__(self, input_channels=3, kernel_size=3, channels=64):
		super(ZSSRNet, self).__init__()
		
		self.conv0 = nn.Conv2d(input_channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		self.conv3 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		self.conv4 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		self.conv5 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		self.conv6 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		self.conv7 = nn.Conv2d(channels, input_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)

		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.relu(self.conv0(x))
		x = self.relu(self.conv1(x))
		x = self.relu(self.conv2(x))
		x = self.relu(self.conv3(x))
		x = self.relu(self.conv4(x))
		x = self.relu(self.conv5(x))
		x = self.relu(self.conv6(x))
		x = self.conv7(x)

		return x


'''CAIR'''
def get_gaussian_kernel(kernel_size=21, sigma=5, channels=3):
    #if not kernel_size: kernel_size = int(2*np.ceil(2*sigma)+1)
    #print("Kernel is: ",kernel_size)
    #print("Sigma is: ",sigma)
    padding = kernel_size//2
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter, padding


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)
        

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
    

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
    

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                            bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                    groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
    

class NAFGroup(nn.Module):
    def __init__(self, n_feats, num_nb=2):
        super().__init__()
        self.body = nn.Sequential(
                        *[NAFBlock(n_feats) for _ in range(num_nb)]
                    )
        self.tail = nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

    def forward(self, inp):
        x = inp

        x = self.body(x)
        x = self.tail(x)
        return x + inp


class ColorAttention(nn.Module):
    def __init__(self, in_channel, num_feat):
        super(ColorAttention, self).__init__()        
        output_nc = num_feat

        num_ng = 2
        num_nb = 2
        kernel_size = 1

        sigma = 12 ## GAUSSIAN_SIGMA

        modules_head = [nn.Conv2d(in_channel, num_feat, kernel_size=kernel_size, padding=0, stride=1, groups=1, bias=True)]

        modules_downsample = [nn.MaxPool2d(kernel_size=2)] 
        self.downsample = nn.Sequential(*modules_downsample)

        modules_body = [
                NAFGroup(num_feat, num_nb) \
                for _ in range(num_ng)
            ]
        
        modules_body.append(nn.Conv2d(num_feat, num_feat, kernel_size, padding=0, stride=1, groups=1, bias=True))

        modules_tail = [nn.Conv2d(num_feat, output_nc, kernel_size, padding=0, stride=1, groups=1, bias=True), nn.Sigmoid()]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        self.blur, self.pad = get_gaussian_kernel(sigma=sigma, channels=in_channel)

    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        x = self.blur(x)
        x = self.head(x)
        x = self.downsample(x)
        x = self.body(x)
        x = self.tail(x)
        return x
    

class CAIR(nn.Module):
    def __init__(self, img_channel=1, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], tta=False):
        super().__init__()
        self.tta = tta
        channel_size = [1, 2, 4]
        self.first_intro = nn.Conv2d(
            in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                            bias=True)
        self.intro = nn.ModuleList()
        self.color_attention = nn.ModuleList()
        for cha in channel_size:
            self.intro.append(
                nn.Conv2d(in_channels=img_channel, out_channels=width*cha, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
            )
            self.color_attention.append(
                ColorAttention(in_channel=img_channel, num_feat=width*cha)
            )
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                            bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, chan, 2, 2)
            )
            chan = chan * 2
        
        chan = chan // 2
        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)],
                nn.Conv2d(chan, chan * 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
            )
        
        chan = chan * 2
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

        self.upsample = Upsample(scale=2, num_feat=width)
        self.conv1x1 = nn.Conv2d(in_channels=width, out_channels=3, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.out_color_attention = ColorAttention(in_channel=img_channel, num_feat=width)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.precision = 'single'

        # self.first_intro.cuda()
    def forward_basic(self, inp):
        B, C, H, W = inp.shape
        datas = []
        inp = self.check_image_size(inp)

        datas.append(self.first_intro(inp))
        res_inp = inp

        for intro, color_attention in zip(self.intro, self.color_attention):
            inp_color = color_attention(inp)
            
            inp = F.interpolate(inp, scale_factor=0.5, mode='bicubic')
            inp_resize = intro(inp)
            inp_color = inp_resize * inp_color
            inp_color = inp_resize + inp_color
            datas.append(inp_color)

        encs = []
        num = [0, 1, 2, 3]
        for encoder, down, data, i in zip(self.encoders, self.downs, datas, num):
            if i == 0:
                x = data
            else:
                x = torch.cat((x, data), 1)
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        color = self.out_color_attention(res_inp)
        color = self.upsample(color)
        color = self.conv1x1(color)
        color = x * color
        x = x + color

        x = x + res_inp

        return x[:, :, :H, :W]

    def forward(self, x):
        self.to(self.device)
        if self.tta:
            return self.forward_x8(x, forward_function=self.forward_basic)
        else:
            return self.forward_basic(x.to(self.device))

    def forward_x8(self, x, forward_function):
        def _transform(v, op):
            if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [x.to(self.device)]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])
        sr_list = [forward_function(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x