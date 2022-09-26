import os
if not ("DISPLAY" in os.environ):
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import util.polarutils
from util.polarutils_torch import calc_normal_from_stokes, calc_admap_from_stokes, calc_dop_from_stokes, calc_fourPolar_from_stokes, calc_polar_phase_from_stokes
from util.util import MAX_8BIT, MAX_16BIT
from matplotlib.colors import LinearSegmentedColormap

aolp_colormap = 'erdc_iceFire'
aolp_colormap_dir = os.path.dirname(os.path.abspath(__file__))
aolp_colormap_data = np.loadtxt(aolp_colormap_dir + "/cmaps/" + aolp_colormap + ".txt")
aolp_cmap = LinearSegmentedColormap.from_list('my_colormap', aolp_colormap_data)

def dolp_colorize(feature, max=255.0, min=0.0):
    feature = (feature - min) / (max - min)
    feature = 255 * plt.cm.turbo(feature)[:, :, :3]
    return feature.astype('uint8')

def aolp_colorize(feature, max=180.0, min=0.0):
    feature = (feature - min) / (max - min)
    feature = 255 * aolp_cmap(feature)[:, :, :3]
    return feature.astype('uint8')

def handle_color(img):
    if img is None:
        return None

    if img.ndim == 2:
        img = np.expand_dims(img, -1)
    if img.shape[2] == 1:
        rgb = np.concatenate((img, img, img), axis=2)
        return rgb
    else:
        return img

def disp_s12(img, max_value=255.0): # [-max_value~max_value]
    if img is None:
        return None
    return np.clip((img + max_value) / 2.0, 0.0, 255.0) # [0~max_value]
    
def disp_admap(img, max_value=255.0):
    if img is None:
        return None
    img[:, :, 0] = (img[:, :, 0] + max_value) / 2.0 # R
    img[:, :, 1] = (img[:, :, 1] + max_value) / 2.0 # G

    return img

def disp_normal(img, max_value=255.0):
    if img is None:
        return None
    img[:, :, 0] = (img[:, :, 0] + max_value) / 2.0 # R
    img[:, :, 1] = (img[:, :, 1] + max_value) / 2.0 # G
    img[:, :, 2] = (img[:, :, 2] + max_value) / 2.0 # B

    return img

def merge_into_row(args, ele, pred):

    disp = {
        'rgb': True,
        'rgb_pred': True,
        'rgb_gt': True,
        's1_sps': False,
        's1_pred': False,
        's1_gt': False,
        's2_sps': False,
        's2_pred': False,
        's2_gt': False,
        'dolp_sps': True,
        'dolp_itp': False,
        'dolp_pred': True,
        'dolp_gt': True,
        'aolp_pred': True,
        'aolp_gt': True,
        'admap_sps': False,
        'admap_itp': False,
        'admap_pred': False,
        'admap_gt': False,
        'pnml_pred': False,
        'pnml_gt': False,
        'dif_pred': False,
        'dif_gt': False,
    }

    if args.vis_dif: # if want to visualize diffuse and specular
        disp['dif_pred'] = True
        disp['dif_gt'] = True

    s12gain = args.s12gain

    img_list = []
    if disp['rgb']:
        s0 = np.squeeze(ele['s0'][0, ...].data.cpu().numpy())
        s0 = np.transpose(s0, (1, 2, 0))
        img_list.append(s0[:, :, ::-1])
    if disp['rgb_pred']:
        s0_pred = pred[0, -3:, ...].data.cpu().numpy()
        s0_pred = np.transpose(s0_pred, (1, 2, 0))
        img_list.append(np.clip(handle_color(s0_pred[:, :, ::-1]), 0, 255))
    if disp['rgb_gt']:
        s0gt = np.squeeze(ele['s0gt'][0, ...].data.cpu().numpy())
        s0gt = np.transpose(s0gt, (1, 2, 0))
        img_list.append(s0gt[:, :, ::-1])
    if disp['s1_sps'] and 's1sps' in ele:
        s1sps = np.squeeze(ele['s1sps'][0, ...].data.cpu().numpy()) * s12gain
        img_list.append(handle_color(disp_s12(s1sps)))
    if disp['s1_pred']:
        s1sps_pred = pred[0, 1, ...].data.cpu().numpy() * s12gain
        img_list.append(handle_color(disp_s12(s1sps_pred)))
    if disp['s1_gt'] and 's1gt' in ele:
        s1gt = ele['s1gt'][0, ...].data.cpu().numpy() * s12gain
        img_list.append(handle_color(disp_s12(s1gt)))
    if disp['s2_sps'] and 's2sps' in ele:
        s2sps = np.squeeze(ele['s2sps'][0, ...].data.cpu().numpy()) * s12gain
        img_list.append(handle_color(disp_s12(s2sps)))
    if disp['s2_pred']:
        s2sps_pred = pred[0, 2, ...].data.cpu().numpy() * s12gain
        img_list.append(handle_color(disp_s12(s2sps_pred)))
    if disp['s2_gt'] and 's2gt' in ele:
        s2gt = ele['s2gt'][0, ...].data.cpu().numpy() * s12gain
        img_list.append(handle_color(disp_s12(s2gt)))
    if disp['dolp_sps']:
        s0sps = ele['s0sps']
        s1sps = ele['s1sps']
        s2sps = ele['s2sps']
        dolpsps = calc_dop_from_stokes(s0sps/MAX_8BIT, s1sps/MAX_8BIT, s2sps/MAX_8BIT)*MAX_8BIT
        dolpsps = np.squeeze(dolpsps[0, ...].data.cpu().numpy())
        dolpsps = handle_color(dolp_colorize(dolpsps))
        img_list.append(dolpsps)
    if disp['dolp_pred']:
        dolppred = calc_dop_from_stokes(pred[0, 0, ...], pred[0, 1, ...], pred[0, 2, ...])*MAX_8BIT
        dolppred = dolppred.data.cpu().numpy()
        img_list.append(handle_color(dolp_colorize(dolppred)))
    if disp['dolp_gt']:
        s0gt = ele['s0gt']
        s1gt = ele['s1gt']
        s2gt = ele['s2gt']
        s0gt_gray = (s0gt[:,0,:,:] + s0gt[:,1,:,:] + s0gt[:,2,:,:])/3.0
        dolpgt = calc_dop_from_stokes(s0gt_gray/MAX_8BIT, s1gt/MAX_8BIT, s2gt/MAX_8BIT)*MAX_8BIT
        dolpgt = np.squeeze(dolpgt[0, ...].data.cpu().numpy())
        img_list.append(handle_color(dolp_colorize(dolpgt)))
    if disp['aolp_pred']:
        aolppred = calc_polar_phase_from_stokes(pred[0, 1, ...], pred[0, 2, ...], radian=False)
        aolppred = aolppred.data.cpu().numpy()
        img_list.append(handle_color(aolp_colorize(aolppred)))
    if disp['aolp_gt']:
        s0gt = ele['s0gt']
        s1gt = ele['s1gt']
        s2gt = ele['s2gt']
        s0gt_gray = (s0gt[:,0,:,:] + s0gt[:,1,:,:] + s0gt[:,2,:,:])/3.0
        aolpgt = calc_polar_phase_from_stokes(s1gt/MAX_8BIT, s2gt/MAX_8BIT, radian=False)
        aolpgt = np.squeeze(aolpgt[0, ...].data.cpu().numpy())
        img_list.append(handle_color(aolp_colorize(aolpgt)))
    if disp['admap_sps']:
        s0sps = ele['s0sps']
        s1sps = ele['s1sps']
        s2sps = ele['s2sps']
        admapsps = calc_admap_from_stokes(s0sps/MAX_8BIT, s1sps/MAX_8BIT, s2sps/MAX_8BIT, forvis=True)*MAX_8BIT
        admapsps = np.squeeze(admapsps[0, ...].data.cpu().numpy())
        admapsps = np.transpose(admapsps, (1, 2, 0))
        img_list.append(disp_admap(admapsps))
    if disp['admap_pred']:
        admappred = calc_admap_from_stokes(pred[0:1, 0, ...], pred[0:1, 1, ...], pred[0:1, 2, ...], forvis=True)*MAX_8BIT
        admappred = np.squeeze(admappred[0, ...].data.cpu().numpy())
        admappred = np.transpose(admappred, (1, 2, 0))
        img_list.append(disp_admap(admappred))
    if disp['admap_gt']:
        s0gt = ele['s0gt']
        s1gt = ele['s1gt']
        s2gt = ele['s2gt']
        s0gt_gray = (s0gt[:,0,:,:] + s0gt[:,1,:,:] + s0gt[:,2,:,:])/3.0
        admapgt = calc_admap_from_stokes(s0gt_gray/MAX_8BIT, s1gt/MAX_8BIT, s2gt/MAX_8BIT, forvis=True)*MAX_8BIT
        admapgt = np.squeeze(admapgt[0, ...].data.cpu().numpy())
        admapgt = np.transpose(admapgt, (1, 2, 0))
        img_list.append(disp_admap(admapgt))
    if disp['pnml_pred']:
        pnmlpred = calc_normal_from_stokes(pred[0:1, 0, ...], pred[0:1, 1, ...], pred[0:1, 2, ...])*MAX_8BIT
        pnmlpred = np.squeeze(pnmlpred[0, ...].data.cpu().numpy())
        pnmlpred = np.transpose(pnmlpred, (1, 2, 0))
        img_list.append(disp_normal(pnmlpred))
    if disp['pnml_gt']:
        s0gt = ele['s0gt']
        s1gt = ele['s1gt']
        s2gt = ele['s2gt']
        s0gt_gray = (s0gt[:,0,:,:] + s0gt[:,1,:,:] + s0gt[:,2,:,:])/3.0
        pnmlgt = calc_normal_from_stokes(s0gt_gray/MAX_8BIT, s1gt/MAX_8BIT, s2gt/MAX_8BIT)*MAX_8BIT
        pnmlgt = np.squeeze(pnmlgt[0, ...].data.cpu().numpy())
        pnmlgt = np.transpose(pnmlgt, (1, 2, 0))
        img_list.append(disp_normal(pnmlgt))
    if disp['dif_pred']:
        s0_pred = pred[0, -3:, ...]
        s0gt_pred_gray = (s0_pred[0,:,:] + s0_pred[1,:,:] + s0_pred[2,:,:])/3.0
        dolppred = calc_dop_from_stokes(pred[0, 0, ...], pred[0, 1, ...], pred[0, 2, ...])*s0gt_pred_gray
        dolppred = dolppred.data.cpu().numpy()
        difpred = s0_pred.data.cpu().numpy() - dolppred
        difpred = np.transpose(difpred, (1, 2, 0))
        img_list.append(np.clip(difpred[:, :, ::-1], 0, 255))
        img_list.append(handle_color(dolppred))
    if disp['dif_gt']:
        s0gt = ele['s0gt']
        s1gt = ele['s1gt']
        s2gt = ele['s2gt']
        s0gt_gray = (s0gt[:,0,:,:] + s0gt[:,1,:,:] + s0gt[:,2,:,:])/3.0
        dolpgt = calc_dop_from_stokes(s0gt_gray/MAX_8BIT, s1gt/MAX_8BIT, s2gt/MAX_8BIT)*s0gt_gray
        dolpgt = np.squeeze(dolpgt[0, ...].data.cpu().numpy())
        difgt = s0gt[0, ...].data.cpu().numpy() - dolpgt
        difgt = np.transpose(difgt, (1, 2, 0))
        img_list.append(np.clip(difgt[:, :, ::-1], 0, 255))
        img_list.append(handle_color(dolpgt))

    img_merge = np.hstack(img_list)
    return img_merge.astype('uint8')


def add_row(img_merge, row):
    return np.vstack([img_merge, row])

def save_image(img_merge, filename):
    image_to_write = cv2.cvtColor(img_merge, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image_to_write)

