from model.basic import *
from torch.nn import functional as F
from model.model_pcn import PCNLight, PCN
from model.model_rgbrn import RGBRN, NoRefine

def select_refine_input_ch(m_refine_input=0):
    if   m_refine_input==0: refine_in_ch = 3 # RGB
    elif m_refine_input==1: refine_in_ch = 4 # RGB + Mask
    elif m_refine_input==2: refine_in_ch = 4 # RGB + S0sps
    elif m_refine_input==3: refine_in_ch = 5 # RGB + S12sps
    elif m_refine_input==4: refine_in_ch = 5 # RGB + S0sps + Mask
    elif m_refine_input==5: refine_in_ch = 6 # RGB + S12sps + Mask
    elif m_refine_input==6: refine_in_ch = 7 # RGB + S0sps + s12sps + Mask
    elif m_refine_input==7: refine_in_ch = 6 # RGB + s0sps + S12sps
    else:                   refine_in_ch = 3
    return refine_in_ch

def select_refine_input(input, m_refine_input=0):
    rgb = input['s0']
    s0sps = input['s0sps']
    s1sps = input['s1sps']
    s2sps = input['s2sps']
    mask = input['mask']

    if m_refine_input==0:
        refine_input = rgb
    elif m_refine_input==1:
        refine_input = torch.cat((rgb, mask), dim=1)
    elif m_refine_input==2:
        refine_input = torch.cat((rgb, s0sps), dim=1)
    elif m_refine_input==3:
        refine_input = torch.cat((rgb, s1sps, s2sps), dim=1)
    elif m_refine_input==4:
        refine_input = torch.cat((rgb, s0sps, mask), dim=1)
    elif m_refine_input==5:
        refine_input = torch.cat((rgb, s1sps, s2sps, mask), dim=1)
    elif m_refine_input==6:
        refine_input = torch.cat((rgb, s0sps, s1sps, s2sps, mask), dim=1)
    elif m_refine_input==7:
        refine_input = torch.cat((rgb, s0sps, s1sps, s2sps), dim=1)
    else:
        refine_input = rgb
    return refine_input    

def select_refine_model(args, refine_in_ch, m_refine_model=0):
    if m_refine_model==0:
        refine = RGBRN(args, in_ch=refine_in_ch, out_ch=3, hid_ch=64, mode=1)
    else:
        refine = NoRefine(args)
    return refine

def select_comp_rgb_ch(m_comp_input_rgb=0):
    if   m_comp_input_rgb==0: comp_rgb_ch = 3 # RGB
    elif m_comp_input_rgb==1: comp_rgb_ch = 3 # Refine
    elif m_comp_input_rgb==2: comp_rgb_ch = 6 # RGB + Refine
    elif m_comp_input_rgb==3: comp_rgb_ch = 3
    elif m_comp_input_rgb==4: comp_rgb_ch = 3
    elif m_comp_input_rgb==5: comp_rgb_ch = 6 # RGB + Refine
    else:                     comp_rgb_ch = 3
    return comp_rgb_ch

def select_comp_input_rgb(input, rgb_refine, m_comp_input_rgb=0, training=True, epoch=0, total_epochs=30):
    rgb = input['s0']
    if m_comp_input_rgb==0:
        comp_input_rgb = rgb
    else:
        comp_input_rgb = rgb_refine
    return comp_input_rgb

def select_comp_extra_ch(m_comp_input_extra=0):
    if m_comp_input_extra==0: comp_extra_ch = 0 # Nothing
    elif m_comp_input_extra==1: comp_extra_ch = 1 # S0sps
    elif m_comp_input_extra==2: comp_extra_ch = 1 # Mask
    elif m_comp_input_extra==3: comp_extra_ch = 2 # S0sps + Mask
    else: comp_extra_ch = 0
    return comp_extra_ch

def select_comp_input_extra(input, m_comp_input_extra=0):
    s0sps = input['s0sps']
    s1sps = input['s1sps']
    s2sps = input['s2sps']
    mask = input['mask']
    if m_comp_input_extra==0:
        comp_input_extra = torch.cat((s1sps, s2sps), dim=1)
    elif m_comp_input_extra==1:
        comp_input_extra = torch.cat((s1sps, s2sps, s0sps), dim=1)
    elif m_comp_input_extra==2:
        comp_input_extra = torch.cat((s1sps, s2sps, mask), dim=1)
    elif m_comp_input_extra==3:
        comp_input_extra = torch.cat((s1sps, s2sps, s0sps, mask), dim=1)
    else:
        comp_input_extra = torch.cat((s1sps, s2sps), dim=1)

    return comp_input_extra

def select_comp_model(args, m_comp_model=0, comp_rgb_ch=3, comp_extra_ch=2):
    if m_comp_model==0:
        comp_model = PCNLight(args, in_ch=2+comp_extra_ch, out_ch=2, rgb_ch=comp_rgb_ch, out_inter=True, pixelshuffle=False)
    else:
        comp_model = PCN(args, in_ch=2+comp_extra_ch, out_ch=2, rgb_ch=comp_rgb_ch, out_inter=True, pixelshuffle=False, mode_afa=2, mode_ftb=3)
    return comp_model

# [args.refine_input] Refine input
# - 0: Only RGB
# - 1: RGB and s0sps
# - 2: RGB and mask
# - 3: RGB and s12sps
# - 4: RGB and s0sps and mask
# - 5: RGB and s12sps and mask
# - 6: RGB and s012sps and mask (default)
# - 7: RGB and s012sps
# [args.refine_model] Refine Model
# - 0: RGB Refinement (default)
# - 1: No Refinement
# [args.comp_input_rgb] Completion Input (RGB)
# - 0: RGB
# - 1: Refine RGB (default)
# [args.comp_input_extra] Completion Input (Extra)
# - 0: Only s12sps
# - 1: s12sps and s0sps
# - 2: s12sps and mask (default)
# - 3: s12sps and s0sps and mask
# [self.args.comp_model] Select Compensation Model
# - 0: Our compensation model
# - 1: Our compensation model (w/ ATA&FTB) (default)
class SNA(nn.Module):
    def __init__(self, args):
        super(SNA, self).__init__()
        self.args = args
        self.m_refine_input = self.args.refine_input
        self.m_refine_model = self.args.refine_model
        self.m_comp_input_rgb = self.args.comp_input_rgb
        self.m_comp_input_extra = self.args.comp_input_extra
        self.m_comp_model = self.args.comp_model

        refine_in_ch = select_refine_input_ch(self.m_refine_input)
        self.refine_model = select_refine_model(self.args, refine_in_ch, self.m_refine_model)

        comp_rgb_ch = select_comp_rgb_ch(self.m_comp_input_rgb)
        comp_extra_ch = select_comp_extra_ch(self.m_comp_input_extra)
        self.comp_model = select_comp_model(self.args, self.m_comp_model, comp_rgb_ch, comp_extra_ch)


    def forward(self, input, epoch=0):

        refine_input = select_refine_input(input, self.m_refine_input)
        rgb_refine = self.refine_model(refine_input)

        comp_input_rgb = select_comp_input_rgb(input, rgb_refine, self.m_comp_input_rgb, self.training, epoch, self.args.epochs)
        comp_input_extra = select_comp_input_extra(input, self.m_comp_input_extra)

        s12 = self.comp_model(comp_input_rgb, comp_input_extra) # 3tuple

        return s12[0], s12[1], s12[2], rgb_refine
