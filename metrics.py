import torch
import math
import numpy as np
from util.util import MAX_8BIT, MAX_16BIT
from util.polarutils_torch import calc_fourPolar_from_stokes, calc_dolp_aolp_admap_from_stokes, calc_normal_from_dop_and_phase
import skimage
skimage_version = float(skimage.__version__[2:])
if skimage_version >= 16.0:
    from skimage.metrics import structural_similarity
else:
    from skimage.measure import compare_ssim as structural_similarity
import cv2

lg_e_10 = math.log(10)

def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / lg_e_10


class Result(object):
    def __init__(self):
        self.mse = {'s0':0., 's1':0., 's2':0., 's012': 0., 's12': 0., '4polar':0., 'dolp':0., 'rgb':0.}
        self.rmse = {'s0':0., 's1':0., 's2':0., 's012': 0., 's12': 0., '4polar':0., 'dolp':0., 'rgb':0.}
        self.mae = {'s0':0., 's1':0., 's2':0., 's012': 0., 's12': 0., '4polar':0., 'dolp':0., 'rgb':0.}
        self.psnr = {'s0':0., 's1':0., 's2':0., 's012': 0., 's12': 0., '4polar':0., 'dolp':0., 'rgb':0.}
        self.delta1 = {'4polar':0., 'dolp':0.}
        self.delta2 = {'4polar':0., 'dolp':0.}
        self.delta3 = {'4polar':0., 'dolp':0.}
        self.diff_angle = {'aolp':0., 'admap':0., 'nml':0}
        self.cos = {'admap':0.}
        self.ssim = {'rgb':0.}
        self.data_time = 0
        self.gpu_time = 0

    def set_to_worst(self):
        self.mse = {'s0':np.inf, 's1':np.inf, 's2':np.inf, 's012':np.inf, 's12': np.inf, '4polar':0., 'dolp':np.inf, 'rgb':np.inf}
        self.rmse = {'s0':np.inf, 's1':np.inf, 's2':np.inf, 's012':np.inf, 's12': np.inf, '4polar':0., 'dolp':np.inf, 'rgb':np.inf}
        self.mae = {'s0':np.inf, 's1':np.inf, 's2':np.inf, 's012':np.inf, 's12': np.inf, '4polar':0., 'dolp':np.inf, 'rgb':np.inf}
        self.psnr = {'s0':np.inf, 's1':np.inf, 's2':np.inf, 's012':np.inf, 's12': np.inf, '4polar':0., 'dolp':np.inf, 'rgb':np.inf}
        self.delta1 = {'4polar':0., 'dolp':0.}
        self.delta2 = {'4polar':0., 'dolp':0.}
        self.delta3 = {'4polar':0., 'dolp':0.}
        self.diff_angle = {'aolp':180., 'admap':180., 'nml':180.}
        self.cos = {'admap':-1.}
        self.ssim = {'rgb':0.}
        self.data_time = 0
        self.gpu_time = 0

    def update(self, mse, rmse, mae, psnr, delta1, delta2, delta3, diff_angle, cos, gpu_time, data_time):
        self.mse = mse
        self.rmse = rmse
        self.mae = mae
        self.psnr = psnr
        self.delta1 = delta1
        self.delta2 = delta2
        self.delta3 = delta3
        self.diff_angle = diff_angle
        self.cos = cos
        self.data_time = data_time
        self.gpu_time = gpu_time

    def calc_psnr(self, mse, R=1.):
        psnr = 10. * np.log10(R**2/np.clip(mse, 0.000001, None))
        return psnr

    def calc_basic_metrics(self, output, target, R=1.):
        abs_diff = (output - target).abs()

        mse = float((torch.pow(abs_diff, 2)).mean())
        rmse = math.sqrt(mse)
        mae = float(abs_diff.mean())
        psnr = self.calc_psnr(mse, R)

        return mse, rmse, mae, psnr

    def calc_delta(self, output, target):

        maxRatio = torch.max(output / target, target / output)
        delta1 = float((maxRatio < 1.25).float().mean())
        delta2 = float((maxRatio < 1.25**2).float().mean())
        delta3 = float((maxRatio < 1.25**3).float().mean())
        return delta1, delta2, delta3

    def calc_aolp_diff_angle(self, output, target): # 0-pi
        abs_diff = (output - target).abs()
        diff_angle = abs_diff / np.pi * 180
        mask = diff_angle > 90
        diff_angle[mask] = 180 - diff_angle[mask]
        return float(diff_angle.mean())

    # dim=1がChannel前提
    def calc_admap_diff_angle(self, output, target):
        cos = (output*target).sum(1, keepdim=True)
        theta = torch.acos(cos.mean()) / np.pi * 180
        return float(theta.mean()), float(cos.mean())

    def calc_nml_diff_angle(self, output, target, target_180):
        cos = (output*target).sum(1, keepdim=True)
        diff_angle = torch.acos(cos.mean()) / np.pi * 180

        cos_180 = (output*target_180).sum(1, keepdim=True)
        diff_angle_180 = torch.acos(cos_180.mean()) / np.pi * 180

        if diff_angle_180 < diff_angle:
            diff_angle = diff_angle_180

        return float(diff_angle.mean())

    def evaluate(self, output, target, output_rgb=None, target_rgb=None):
        output /= MAX_8BIT
        target /= MAX_8BIT

        output_s12 = output[:,1:,:,:] # b 2 h w
        target_s12 = target[:,1:,:,:]
        output_s0 = output[:,0,:,:] # b h w
        target_s0 = target[:,0,:,:]
        output_s1 = output[:,1,:,:]
        target_s1 = target[:,1,:,:]
        output_s2 = output[:,2,:,:]
        target_s2 = target[:,2,:,:]

        valid_mask_s012 = torch.ones_like(output, dtype=bool)
        output_s012 = output[valid_mask_s012]
        target_s012 = target[valid_mask_s012]

        valid_mask_s12 = torch.ones_like(output_s12, dtype=bool)
        output_s12 = output_s12[valid_mask_s12]
        target_s12 = target_s12[valid_mask_s12]

        valid_mask = torch.ones_like(output_s1, dtype=bool)
        output_s0 = output_s0[valid_mask]  # b*h*w
        target_s0 = target_s0[valid_mask]
        output_s1 = output_s1[valid_mask]
        target_s1 = target_s1[valid_mask]
        output_s2 = output_s2[valid_mask]
        target_s2 = target_s2[valid_mask]

        output_i0, output_i45, output_i90, output_i135 = calc_fourPolar_from_stokes(output_s0, output_s1, output_s2)
        target_i0, target_i45, target_i90, target_i135 = calc_fourPolar_from_stokes(target_s0, target_s1, target_s2)
        output_4polar = torch.stack([output_i0, output_i45, output_i90, output_i135], axis=1)
        target_4polar = torch.stack([target_i0, target_i45, target_i90, target_i135], axis=1)

        output_dolp, output_aolp, output_admap = calc_dolp_aolp_admap_from_stokes(output_s0, output_s1, output_s2)
        target_dolp, target_aolp, target_admap = calc_dolp_aolp_admap_from_stokes(target_s0, target_s1, target_s2)

        output_nml = calc_normal_from_dop_and_phase(output_dolp, output_aolp, radian_aolp=True)
        target_nml = calc_normal_from_dop_and_phase(target_dolp, target_aolp, radian_aolp=True)
        target_nml_180 = calc_normal_from_dop_and_phase(target_dolp, target_aolp+np.pi, radian_aolp=True)

        self.mse['s012'], self.rmse['s012'], self.mae['s012'], self.psnr['s012'] = self.calc_basic_metrics(output_s012, target_s012, 2.)
        self.mse['s12'], self.rmse['s12'], self.mae['s12'], self.psnr['s12'] = self.calc_basic_metrics(output_s12, target_s12, 2.)
        self.mse['s0'], self.rmse['s0'], self.mae['s0'], self.psnr['s0'] = self.calc_basic_metrics(output_s0, target_s0, 1.)
        self.mse['s1'], self.rmse['s1'], self.mae['s1'], self.psnr['s1'] = self.calc_basic_metrics(output_s1, target_s1, 2.)
        self.mse['s2'], self.rmse['s2'], self.mae['s2'], self.psnr['s2'] = self.calc_basic_metrics(output_s2, target_s2, 2.)
        self.mse['4polar'], self.rmse['4polar'], self.mae['4polar'], self.psnr['4polar'] = self.calc_basic_metrics(output_4polar, target_4polar, 1.)
        self.mse['dolp'], self.rmse['dolp'], self.mae['dolp'], self.psnr['dolp'] = self.calc_basic_metrics(output_dolp, target_dolp, 1.)

        self.delta1['4polar'], self.delta2['4polar'], self.delta3['4polar'] = self.calc_delta(output_4polar, target_4polar)
        self.delta1['dolp'], self.delta2['dolp'], self.delta3['dolp'] = self.calc_delta(output_dolp, target_dolp)

        self.diff_angle['aolp'] = self.calc_aolp_diff_angle(output_aolp, target_aolp)
        self.diff_angle['admap'], self.cos['admap'] =self.calc_admap_diff_angle(output_admap, target_admap)
        self.diff_angle['nml'] = self.calc_nml_diff_angle(output_nml, target_nml, target_nml_180)

        if output_rgb is not None and target_rgb is not None:
            valid_mask_rgb = torch.ones_like(output_rgb, dtype=bool)
            output_rgb_flat = output_rgb[valid_mask_rgb] / MAX_8BIT
            target_rgb_flat = target_rgb[valid_mask_rgb] / MAX_8BIT
            self.mse['rgb'], self.rmse['rgb'], self.mae['rgb'], self.psnr['rgb'] = self.calc_basic_metrics(output_rgb_flat, target_rgb_flat, 1.)

            bs = output_rgb.shape[0]
            for i in range(bs):
                src = output_rgb[i, :, :, :].data.cpu().numpy()
                gt = target_rgb[i, :, :, :].data.cpu().numpy()
                src = np.transpose(src, (1, 2, 0))
                gt = np.transpose(gt, (1, 2, 0))
                self.ssim['rgb'] = structural_similarity(src.astype(np.int16), gt.astype(np.int16), data_range=int(MAX_8BIT), multichannel=True)

        self.data_time = 0
        self.gpu_time = 0


class AverageMeter(object):
    def __init__(self):
        self.reset(time_stable=True)

    def reset(self, time_stable):
        self.count = 0.0
        self.sum_mse = {'s0':0., 's1':0., 's2':0., 's012': 0., 's12': 0., '4polar':0., 'dolp':0., 'rgb':0.}
        self.sum_rmse = {'s0':0., 's1':0., 's2':0., 's012': 0., 's12': 0., '4polar':0., 'dolp':0., 'rgb':0.}
        self.sum_mae = {'s0':0., 's1':0., 's2':0., 's012': 0., 's12': 0., '4polar':0., 'dolp':0., 'rgb':0.}
        self.sum_psnr = {'s0':0., 's1':0., 's2':0., 's012': 0., 's12': 0., '4polar':0., 'dolp':0., 'rgb':0.}
        self.sum_delta1 = {'4polar':0., 'dolp':0.}
        self.sum_delta2 = {'4polar':0., 'dolp':0.}
        self.sum_delta3 = {'4polar':0., 'dolp':0.}
        self.sum_diff_angle = {'aolp':0., 'admap':0., 'nml':0.}
        self.sum_cos = {'admap':0.}
        self.sum_ssim = {'rgb':0.}

        self.sum_data_time = 0
        self.sum_gpu_time = 0
        self.time_stable = time_stable
        self.time_stable_counter_init = 10
        self.time_stable_counter = self.time_stable_counter_init

    def update(self, result, gpu_time, data_time, n=1):
        self.count += n

        for k, v in result.mse.items(): self.sum_mse[k] += n * v        
        for k, v in result.rmse.items(): self.sum_rmse[k] += n * v        
        for k, v in result.mae.items(): self.sum_mae[k] += n * v        
        for k, v in result.psnr.items(): self.sum_psnr[k] += n * v        
        for k, v in result.delta1.items(): self.sum_delta1[k] += n * v        
        for k, v in result.delta2.items(): self.sum_delta2[k] += n * v        
        for k, v in result.delta3.items(): self.sum_delta3[k] += n * v        
        for k, v in result.diff_angle.items(): self.sum_diff_angle[k] += n * v        
        for k, v in result.cos.items(): self.sum_cos[k] += n * v        
        for k, v in result.ssim.items(): self.sum_ssim[k] += n * v        

        self.sum_data_time += n * data_time
        if self.time_stable == True and self.time_stable_counter > 0:
            self.time_stable_counter = self.time_stable_counter - 1
        else:
            self.sum_gpu_time += n * gpu_time

    def average(self):
        avg = Result()

        for k, v in self.sum_mse.items():
            avg.mse[k] = v / self.count
        for k, v in self.sum_rmse.items():
            avg.rmse[k] = v / self.count
        for k, v in self.sum_mae.items():
            avg.mae[k] = v / self.count
        for k, v in self.sum_psnr.items():
            avg.psnr[k] = v / self.count
        for k, v in self.sum_delta1.items():
            avg.delta1[k] = v / self.count
        for k, v in self.sum_delta2.items():
            avg.delta2[k] = v / self.count
        for k, v in self.sum_delta3.items():
            avg.delta3[k] = v / self.count
        for k, v in self.sum_diff_angle.items():
            avg.diff_angle[k] = v / self.count
        for k, v in self.sum_cos.items():
            avg.cos[k] = v / self.count
        for k, v in self.sum_ssim.items():
            avg.ssim[k] = v / self.count
        avg.data_time = self.sum_data_time / self.count

        if self.time_stable == True:
            if self.count > 0 and self.count - self.time_stable_counter_init > 0:
                avg.gpu_time = self.sum_gpu_time / (self.count - self.time_stable_counter_init)
            elif self.count > 0:
                avg.gpu_time = 0
        elif self.count > 0:
            avg.gpu_time = self.sum_gpu_time / self.count
        return avg
