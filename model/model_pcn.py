from model.basic import *
from torch.nn import functional as F

class PCNLight(nn.Module): # No FTB and AFA
    def __init__(self, args, in_ch=2, out_ch=2, rgb_ch=3, out_inter=False, pixelshuffle=False):
        super(PCNLight, self).__init__()
        self.args = args
        self.geoplanes = 0
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.out_inter = out_inter

        # s0 encoder
        self.s0_conv_init = convbnrelu(in_channels=in_ch+rgb_ch, out_channels=32, kernel_size=5, stride=1, padding=2)

        self.s0_encoder_layer1 = EncoderBlock(inplanes=32, planes=64, stride=2, geoplanes=self.geoplanes)
        self.s0_encoder_layer2 = EncoderBlock(inplanes=64, planes=64, stride=1, geoplanes=self.geoplanes)
        self.s0_encoder_layer3 = EncoderBlock(inplanes=64, planes=128, stride=2, geoplanes=self.geoplanes)
        self.s0_encoder_layer4 = EncoderBlock(inplanes=128, planes=128, stride=1, geoplanes=self.geoplanes)
        self.s0_encoder_layer5 = EncoderBlock(inplanes=128, planes=256, stride=2, geoplanes=self.geoplanes)
        self.s0_encoder_layer6 = EncoderBlock(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)
        self.s0_encoder_layer7 = EncoderBlock(inplanes=256, planes=512, stride=2, geoplanes=self.geoplanes)
        self.s0_encoder_layer8 = EncoderBlock(inplanes=512, planes=512, stride=1, geoplanes=self.geoplanes)
        self.s0_encoder_layer9 = EncoderBlock(inplanes=512, planes=1024, stride=2, geoplanes=self.geoplanes)
        self.s0_encoder_layer10 = EncoderBlock(inplanes=1024, planes=1024, stride=1, geoplanes=self.geoplanes)

        self.s0_decoder_layer8 = deconvbnrelu(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1, pixelshuffle=pixelshuffle)
        self.s0_decoder_layer6 = deconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1, pixelshuffle=pixelshuffle)
        self.s0_decoder_layer4 = deconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1, pixelshuffle=pixelshuffle)
        self.s0_decoder_layer2 = deconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1, pixelshuffle=pixelshuffle)
        self.s0_decoder_layer0 = deconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1, pixelshuffle=pixelshuffle)
        self.s0_decoder_output = deconvbn(in_channels=32, out_channels=out_ch+1, kernel_size=3, stride=1, padding=1, output_padding=0, pixelshuffle=False)
        self.s0_decoder_conf = relu()


        # s12 encoder
        self.s12_conv_init = convbnrelu(in_channels=in_ch+out_ch, out_channels=32, kernel_size=5, stride=1, padding=2)

        self.s12_layer1 = EncoderBlock(inplanes=32, planes=64, stride=2, geoplanes=self.geoplanes)
        self.s12_layer2 = EncoderBlock(inplanes=64, planes=64, stride=1, geoplanes=self.geoplanes)
        self.s12_layer3 = EncoderBlock(inplanes=128, planes=128, stride=2, geoplanes=self.geoplanes)
        self.s12_layer4 = EncoderBlock(inplanes=128, planes=128, stride=1, geoplanes=self.geoplanes)
        self.s12_layer5 = EncoderBlock(inplanes=256, planes=256, stride=2, geoplanes=self.geoplanes)
        self.s12_layer6 = EncoderBlock(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)
        self.s12_layer7 = EncoderBlock(inplanes=512, planes=512, stride=2, geoplanes=self.geoplanes)
        self.s12_layer8 = EncoderBlock(inplanes=512, planes=512, stride=1, geoplanes=self.geoplanes)
        self.s12_layer9 = EncoderBlock(inplanes=1024, planes=1024, stride=2, geoplanes=self.geoplanes)
        self.s12_layer10 = EncoderBlock(inplanes=1024, planes=1024, stride=1, geoplanes=self.geoplanes)

        # decoder
        self.decoder_layer1 = deconvbnrelu(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1, pixelshuffle=pixelshuffle)
        self.decoder_layer2 = deconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1, pixelshuffle=pixelshuffle)
        self.decoder_layer3 = deconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1, pixelshuffle=pixelshuffle)
        self.decoder_layer4 = deconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1, pixelshuffle=pixelshuffle)
        self.decoder_layer5 = deconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1, pixelshuffle=pixelshuffle)
        self.decoder_layer6 = convbn(in_channels=32, out_channels=out_ch+1, kernel_size=3, stride=1, padding=1)
        self.decoder_conf = relu()

        self.softmax = nn.Softmax(dim=1)
        self.pooling = nn.AvgPool2d(kernel_size=2)

        weights_init(self)

    def forward(self, rgb, x):
        # 1st branch----------------------
        s0_feature = self.s0_conv_init(torch.cat((rgb, x), dim=1)) # b 3+in_ch h w -> b 32 h w

        # Geometric Convolution
        t = self.s0_encoder_layer1(s0_feature) # b 32 h w -> b 64 h/2 w/2
        s0_feature2 = self.s0_encoder_layer2(t) # b 64 h/2 w/2
        t = self.s0_encoder_layer3(s0_feature2) # b 64 h/2 w/2 -> b 128 h/4 w/4
        s0_feature4 = self.s0_encoder_layer4(t) # b 128 h/4 w/4
        t = self.s0_encoder_layer5(s0_feature4) # b 128 h/4 w/4 -> b 256 h/8 w/8
        s0_feature6 = self.s0_encoder_layer6(t) # b 256 h/8 w/8
        t = self.s0_encoder_layer7(s0_feature6) # b 256 h/8 w/8 -> b 512 h/16 w/16
        s0_feature8 = self.s0_encoder_layer8(t) # b 512 h/16 w/16
        t = self.s0_encoder_layer9(s0_feature8) # b 512 h/16 w/16 -> b 1024 h/32 w/32
        s0_feature10 = self.s0_encoder_layer10(t) # b 1024 h/32 w/32

        # decoder:convTrans->BatchNorm->Relu
        s0_feature8 = self.s0_decoder_layer8(s0_feature10) + s0_feature8 # b 512 h/16 w/16
        s0_feature6 = self.s0_decoder_layer6(s0_feature8) + s0_feature6 # b 256 h/8 w/8
        s0_feature4 = self.s0_decoder_layer4(s0_feature6) + s0_feature4 # b 128 h/4 w/4
        s0_feature2 = self.s0_decoder_layer2(s0_feature4) + s0_feature2   # b 64 h/2 w/2
        s0_feature0 = self.s0_decoder_layer0(s0_feature2) + s0_feature # b 32 h w

        output_1st = self.s0_decoder_output(s0_feature0) # b 32 h w -> b out_ch+1 h w
        y_1st = output_1st[:, 0:self.out_ch, :, :] # b out_ch h w
        conf_1st = self.s0_decoder_conf(output_1st[:, self.out_ch:, :, :])

        # 2nd branch----------------------
        t = self.s12_conv_init(torch.cat((x, y_1st), dim=1)) # b 2 h w -> b 32 h w
        t = self.s12_layer1(t) # b 32 h w -> 64 h/2 w/2
        sparsed_feature2 = self.s12_layer2(t) # b 64 h/2 w/2

        t = torch.cat([s0_feature2, sparsed_feature2], 1) # b 64 h/2 w/2 -> b 128 h/2 w/2
        t = self.s12_layer3(t) # b 128 h/2 w/2 -> b 128 h/4 w/4
        sparsed_feature4 = self.s12_layer4(t) # b 128 h/4 w/4

        t = torch.cat([s0_feature4, sparsed_feature4], 1) # b 128 h/4 w/4 -> b 256 h/4 w/4 
        t = self.s12_layer5(t) # b 256 h/4 w/4 -> b 256 h/8 w/8
        sparsed_feature6 = self.s12_layer6(t) # b b 256 h/8 w/8

        t = torch.cat([s0_feature6, sparsed_feature6], 1) # b 256 h/8 w/8 -> b 512 h/8 w/8
        t = self.s12_layer7(t) # b 512 h/8 w/8 -> b 512 h/16 w/16
        sparsed_feature8 = self.s12_layer8(t) # b 512 h/16 w/16

        t = torch.cat([s0_feature8, sparsed_feature8], 1) # b 512 h/16 w/16 -> b 1024 h/16 w/16
        t = self.s12_layer9(t) # b 1024 h/16 w/16 -> b 1024 h/32 w/32
        t = self.s12_layer10(t) # 1024 h/32 w/32

        # -----------------------------------------------------------------------------------------

        t = s0_feature10 + t # b 1024 h/32 w/32
        t = self.decoder_layer1(t) # b 1024 h/32 w/32 -> b 512 h/16 w/16

        t = sparsed_feature8 + t # b 512 h/16 w/16
        t = self.decoder_layer2(t) # b 512 h/16 w/16 -> b 256 h/8 w/8

        t = sparsed_feature6 + t # b 256 h/8 w/8
        t = self.decoder_layer3(t) # b 256 h/8 w/8 -> b 128 h/4 w/4

        t = sparsed_feature4 + t # b 128 h/4 w/4
        t = self.decoder_layer4(t) # b 128 h/4 w/4 -> b 64 h/2 w/2

        t = sparsed_feature2 + t # b 64 h/2 w/2
        t = self.decoder_layer5(t) # b 64 h/2 w/2 -> b 32 h w

        output_2nd = self.decoder_layer6(t) # b 32 h w -> b 2 h w
        y_2nd = output_2nd[:, 0:self.out_ch, :, :] # b 2 h w
        conf_2nd = self.decoder_conf(output_2nd[:, self.out_ch:, :, :])

        # SoftMax two confidence levels and blend s12 according to the confidence level
        conf_1st, conf_2nd = torch.chunk(self.softmax(torch.cat((conf_1st, conf_2nd), dim=1)), 2, dim=1)
        y = conf_1st*y_1st + conf_2nd*y_2nd

        if self.out_inter:
            return y_1st, y_2nd, y
        else:
            return y


class PCN(nn.Module):
    def __init__(self, args, in_ch=2, out_ch=2, rgb_ch=3, out_inter=False, pixelshuffle=False, mode_afa=0, mode_ftb=0):
        super(PCN, self).__init__()
        self.args = args
        self.geoplanes = 0
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.out_inter = out_inter
        self.mode_afa = mode_afa
        self.mode_ftb = mode_ftb

        # s0 encoder
        self.s0_conv_init = convbnrelu(in_channels=in_ch+rgb_ch, out_channels=32, kernel_size=5, stride=1, padding=2)

        self.s0_encoder_layer1 = EncoderBlock(inplanes=32, planes=64, stride=2, geoplanes=self.geoplanes)
        self.s0_encoder_layer2 = EncoderBlock(inplanes=64, planes=64, stride=1, geoplanes=self.geoplanes)
        self.s0_encoder_layer3 = EncoderBlock(inplanes=64, planes=128, stride=2, geoplanes=self.geoplanes)
        self.s0_encoder_layer4 = EncoderBlock(inplanes=128, planes=128, stride=1, geoplanes=self.geoplanes)
        self.s0_encoder_layer5 = EncoderBlock(inplanes=128, planes=256, stride=2, geoplanes=self.geoplanes)
        self.s0_encoder_layer6 = EncoderBlock(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)
        self.s0_encoder_layer7 = EncoderBlock(inplanes=256, planes=512, stride=2, geoplanes=self.geoplanes)
        self.s0_encoder_layer8 = EncoderBlock(inplanes=512, planes=512, stride=1, geoplanes=self.geoplanes)
        self.s0_encoder_layer9 = EncoderBlock(inplanes=512, planes=1024, stride=2, geoplanes=self.geoplanes)
        self.s0_encoder_layer10 = EncoderBlock(inplanes=1024, planes=1024, stride=1, geoplanes=self.geoplanes)

        self.s0_decoder_layer8 = deconvbnrelu(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1, pixelshuffle=pixelshuffle)
        self.s0_decoder_layer6 = deconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1, pixelshuffle=pixelshuffle)
        self.s0_decoder_layer4 = deconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1, pixelshuffle=pixelshuffle)
        self.s0_decoder_layer2 = deconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1, pixelshuffle=pixelshuffle)
        self.s0_decoder_layer0 = deconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1, pixelshuffle=pixelshuffle)
        self.s0_decoder_output = deconvbn(in_channels=32, out_channels=out_ch+1, kernel_size=3, stride=1, padding=1, output_padding=0, pixelshuffle=False)
        self.s0_decoder_conf = relu()


        # s12 encoder
        self.s12_conv_init = convbnrelu(in_channels=in_ch+out_ch, out_channels=32, kernel_size=5, stride=1, padding=2)

        self.s12_layer1 = EncoderBlock(inplanes=32, planes=64, stride=2, geoplanes=self.geoplanes)
        self.s12_layer2 = EncoderBlock(inplanes=64, planes=64, stride=1, geoplanes=self.geoplanes)
        self.s12_layer3 = EncoderBlock(inplanes=128, planes=128, stride=2, geoplanes=self.geoplanes)
        self.s12_layer4 = EncoderBlock(inplanes=128, planes=128, stride=1, geoplanes=self.geoplanes)
        self.s12_layer5 = EncoderBlock(inplanes=256, planes=256, stride=2, geoplanes=self.geoplanes)
        self.s12_layer6 = EncoderBlock(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)
        self.s12_layer7 = EncoderBlock(inplanes=512, planes=512, stride=2, geoplanes=self.geoplanes)
        self.s12_layer8 = EncoderBlock(inplanes=512, planes=512, stride=1, geoplanes=self.geoplanes)
        self.s12_layer9 = EncoderBlock(inplanes=1024, planes=1024, stride=2, geoplanes=self.geoplanes)
        self.s12_layer10 = EncoderBlock(inplanes=1024, planes=1024, stride=1, geoplanes=self.geoplanes)

        # decoder
        self.decoder_layer1 = deconvbnrelu(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1, pixelshuffle=pixelshuffle)
        self.decoder_layer2 = deconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1, pixelshuffle=pixelshuffle)
        self.decoder_layer3 = deconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1, pixelshuffle=pixelshuffle)
        self.decoder_layer4 = deconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1, pixelshuffle=pixelshuffle)
        self.decoder_layer5 = deconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1, pixelshuffle=pixelshuffle)
        self.decoder_layer6 = convbn(in_channels=32, out_channels=out_ch+1, kernel_size=3, stride=1, padding=1)
        self.decoder_conf = relu()

        self.afa_s0_8 = AFA(inplanes=512)
        self.afa_s0_6 = AFA(inplanes=256)
        self.afa_s0_4 = AFA(inplanes=128)
        self.afa_s0_2 = AFA(inplanes=64)
        self.afa_s0 = AFA(inplanes=32)

        self.ftb_c_8 = FTB(inplanes=512)
        self.ftb_c_6 = FTB(inplanes=256)
        self.ftb_c_4 = FTB(inplanes=128)
        self.ftb_c_2 = FTB(inplanes=64)

        self.softmax = nn.Softmax(dim=1)
        self.pooling = nn.AvgPool2d(kernel_size=2)

        weights_init(self)

    def forward(self, rgb, x):
        # 1st branch----------------------
        s0_feature = self.s0_conv_init(torch.cat((rgb, x), dim=1)) # b 3+in_ch h w -> b 32 h w

        # Geometric Convolution
        t = self.s0_encoder_layer1(s0_feature) # b 32 h w -> b 64 h/2 w/2
        s0_feature2 = self.s0_encoder_layer2(t) # b 64 h/2 w/2
        t = self.s0_encoder_layer3(s0_feature2) # b 64 h/2 w/2 -> b 128 h/4 w/4
        s0_feature4 = self.s0_encoder_layer4(t) # b 128 h/4 w/4
        t = self.s0_encoder_layer5(s0_feature4) # b 128 h/4 w/4 -> b 256 h/8 w/8
        s0_feature6 = self.s0_encoder_layer6(t) # b 256 h/8 w/8
        t = self.s0_encoder_layer7(s0_feature6) # b 256 h/8 w/8 -> b 512 h/16 w/16
        s0_feature8 = self.s0_encoder_layer8(t) # b 512 h/16 w/16
        t = self.s0_encoder_layer9(s0_feature8) # b 512 h/16 w/16 -> b 1024 h/32 w/32
        s0_feature10 = self.s0_encoder_layer10(t) # b 1024 h/32 w/32

        # decoder: convTrans->BatchNorm->Relu
        s0_feature8 = self.afa_s0_8(self.s0_decoder_layer8(s0_feature10), s0_feature8) # b 512 h/16 w/16
        s0_feature6 = self.afa_s0_6(self.s0_decoder_layer6(s0_feature8), s0_feature6) # b 256 h/8 w/8
        s0_feature4 = self.afa_s0_4(self.s0_decoder_layer4(s0_feature6), s0_feature4) # b 128 h/4 w/4
        s0_feature2 = self.afa_s0_2(self.s0_decoder_layer2(s0_feature4), s0_feature2)   # b 64 h/2 w/2
        s0_feature0 = self.afa_s0(self.s0_decoder_layer0(s0_feature2), s0_feature) # b 32 h w

        output_1st = self.s0_decoder_output(s0_feature0) # b 32 h w -> b out_ch+1 h w
        y_1st = output_1st[:, 0:self.out_ch, :, :] # b out_ch h w
        conf_1st = self.s0_decoder_conf(output_1st[:, self.out_ch:, :, :])

        # 2nd branch----------------------
        t = self.s12_conv_init(torch.cat((x, y_1st), dim=1)) # b 2 h w -> b 32 h w

        t = self.s12_layer1(t) # b 32 h w -> 64 h/2 w/2
        sparsed_feature2 = self.s12_layer2(t) # b 64 h/2 w/2


        t = torch.cat([self.ftb_c_2(s0_feature2), sparsed_feature2], 1) # b 64 h/2 w/2 -> b 128 h/2 w/2
        t = self.s12_layer3(t) # b 128 h/2 w/2 -> b 128 h/4 w/4
        sparsed_feature4 = self.s12_layer4(t) # b 128 h/4 w/4

        t = torch.cat([self.ftb_c_4(s0_feature4), sparsed_feature4], 1) # b 128 h/4 w/4 -> b 256 h/4 w/4 
        t = self.s12_layer5(t) # b 256 h/4 w/4 -> b 256 h/8 w/8
        sparsed_feature6 = self.s12_layer6(t) # b b 256 h/8 w/8

        t = torch.cat([self.ftb_c_6(s0_feature6), sparsed_feature6], 1) # b 256 h/8 w/8 -> b 512 h/8 w/8
        t = self.s12_layer7(t) # b 512 h/8 w/8 -> b 512 h/16 w/16
        sparsed_feature8 = self.s12_layer8(t) # b 512 h/16 w/16

        t = torch.cat([self.ftb_c_8(s0_feature8), sparsed_feature8], 1) # b 512 h/16 w/16 -> b 1024 h/16 w/16

        t = self.s12_layer9(t) # b 1024 h/16 w/16 -> b 1024 h/32 w/32
        t = self.s12_layer10(t) # 1024 h/32 w/32

        # -----------------------------------------------------------------------------------------

        t = s0_feature10 + t # b 1024 h/32 w/32
        t = self.decoder_layer1(t) # b 1024 h/32 w/32 -> b 512 h/16 w/16

        t = sparsed_feature8 + t # b 512 h/16 w/16
        t = self.decoder_layer2(t) # b 512 h/16 w/16 -> b 256 h/8 w/8

        t = sparsed_feature6 + t # b 256 h/8 w/8
        t = self.decoder_layer3(t) # b 256 h/8 w/8 -> b 128 h/4 w/4

        t = sparsed_feature4 + t # b 128 h/4 w/4
        t = self.decoder_layer4(t) # b 128 h/4 w/4 -> b 64 h/2 w/2

        t = sparsed_feature2 + t # b 64 h/2 w/2
        t = self.decoder_layer5(t) # b 64 h/2 w/2 -> b 32 h w

        output_2nd = self.decoder_layer6(t) # b 32 h w -> b 2 h w
        y_2nd = output_2nd[:, 0:self.out_ch, :, :] # b 2 h w
        conf_2nd = self.decoder_conf(output_2nd[:, self.out_ch:, :, :])

        # SoftMax two confidence levels and blend s12 according to the confidence level
        conf_1st, conf_2nd = torch.chunk(self.softmax(torch.cat((conf_1st, conf_2nd), dim=1)), 2, dim=1)
        y = conf_1st*y_1st + conf_2nd*y_2nd

        if self.out_inter:
            return y_1st, y_2nd, y
        else:
            return y
