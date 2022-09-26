import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def convbnrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(inplace=True)
	)

def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
		nn.ReLU(inplace=True)
	)

def deconvbnrelu(in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1, pixelshuffle=False):
    if pixelshuffle:
        return nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(int(in_channels/4), out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


def convbn(in_channels, out_channels, kernel_size=3,stride=1, padding=1):
    return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
		nn.BatchNorm2d(out_channels)
	)

def deconvbn(in_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=0, pixelshuffle=False):
    if pixelshuffle:
        return nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(int(in_channels/4), out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )

def relu():
    return nn.ReLU(inplace=True)

def conv9x9(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False, padding=4):
    """9x9 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=9, stride=stride,
                     padding=padding, groups=groups, bias=bias, dilation=dilation)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False, padding=1):
    """3x3 convolution with padding"""
    if padding >= 1:
        padding = dilation
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=bias, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1, groups=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, bias=bias)

class EncoderBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, geoplanes=3):
        super(EncoderBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes + geoplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes+geoplanes, planes)
        self.bn2 = norm_layer(planes)
        if stride != 1 or inplanes != planes:
            # If stride is greater than 1, or if inplane and plane are different, downsample
            downsample = nn.Sequential(
                conv1x1(inplanes+geoplanes, planes, stride),
                norm_layer(planes),
            )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, g1=None, g2=None):
        identity = x
        if g1 is not None:
            x = torch.cat((x, g1), 1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if g2 is not None:
            out = torch.cat((g2,out), 1)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class FTB(nn.Module):
    def __init__(self, inplanes, stride=1, dilation=1):
        super(FTB, self).__init__()

        hidplanes = inplanes
        outplanes = inplanes

        self.conv1 = conv1x1(inplanes, outplanes, stride=stride)
        self.conv2 = conv3x3(outplanes, hidplanes, stride=stride, dilation=dilation)
        self.conv3 = conv3x3(hidplanes, outplanes, stride=stride, dilation=dilation)
        self.relu = nn.ReLU(inplace=False)
        self.bn = nn.BatchNorm2d(hidplanes)

    def forward(self, x):
        x = self.conv1(x)
        t = self.conv2(x)
        t = self.bn(t)
        t = self.relu(t)
        t = self.conv3(t)
        x = t + x
        x = self.relu(x)
        return x

class AFA(nn.Module):
    def __init__(self, inplanes, stride=1, dilation=1, lastrelu=False):
        super(AFA, self).__init__()

        hidplanes = inplanes*2
        outplanes = inplanes

        self.fc1 = nn.Linear(inplanes*2, hidplanes)
        self.fc2 = nn.Linear(hidplanes, outplanes)

        self.relu = nn.ReLU(inplace=False)
        self.bn = nn.BatchNorm2d(hidplanes)
        self.sigmoid = nn.Sigmoid()
        self.lastrelu = lastrelu


    def forward(self, hf, lf):
        att = torch.cat((lf, hf), dim=1) # b, c*2, h, w
        att = att.mean([2, 3]) # b, l_c+h_c
        att = self.fc1(att)
        att = self.relu(att)
        att = self.fc2(att)
        if self.lastrelu: att = self.relu(att)
        att = self.sigmoid(att)

        lf = lf*att.unsqueeze(2).unsqueeze(3)

        return lf + hf