from model.basic import *
from torch.nn import functional as F

class RefineBlock(nn.Module):
    def __init__(self, inplanes, hidplanes, outplanes, stride=1, dilation=1):
        super(RefineBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, hidplanes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(hidplanes)
        self.conv2 = conv3x3(hidplanes, outplanes, stride=stride, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out + x

class LastBlock(nn.Module):
    def __init__(self, inplanes, hidplanes, outplanes, stride=1, dilation=1, mode=0):
        super(LastBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, hidplanes, stride=stride, dilation=dilation)
        self.conv2 = conv3x3(hidplanes, hidplanes, stride=stride, dilation=dilation)
        self.conv3 = conv9x9(hidplanes, outplanes, stride=stride, dilation=dilation)
        self.relu = nn.ReLU(inplace=False)
        self.mode = mode

    def forward(self, x, rgb):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        if self.mode == 1: x = rgb + x
        x = self.relu(x)

        return x

class NoRefine(nn.Module):
    def __init__(self, args):
        super(NoRefine, self).__init__()
        self.args = args

    def forward(self, x):
        return x


class RGBRN(nn.Module):
    def __init__(self, args, in_ch=3, out_ch=3, hid_ch=64, mode=0):
        super(RGBRN, self).__init__()
        self.args = args
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.hid_ch = hid_ch

        # s0 encoder
        self.conv_init = convbnrelu(in_ch, hid_ch, kernel_size=9, padding=4)
        self.refblock1 = RefineBlock(hid_ch, hid_ch, hid_ch)
        self.refblock2 = RefineBlock(hid_ch, hid_ch, hid_ch)
        self.refblock3 = RefineBlock(hid_ch, hid_ch, hid_ch)
        self.refblock4 = RefineBlock(hid_ch, hid_ch, hid_ch)
        self.conv_last = LastBlock(hid_ch, hid_ch, out_ch, mode=mode)

        weights_init(self)

    def forward(self, x):
        out = self.conv_init(x)
        out = self.refblock1(out)
        out = self.refblock2(out)
        out = self.refblock3(out)
        out = self.refblock4(out)
        out = self.conv_last(out, x[:,:3,...]) # Slice RGB
        return out