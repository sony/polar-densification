"""
These functions deal with polarization information.
"""
import numpy as np
import torch

PHASEMAX_DEG = 180.
ZENITHMAX_DEG = 90.
REF_ID = 1.5

def rad2deg(rad):
    deg = rad / np.pi * 180.0
    return deg

def calc_fourPolar_from_stokes(s0, s1, s2):
    """
    This function calculates the polarization images of (0[deg], 45[deg], 90[deg], 135[deg])
    by using input s0, s1, s2 images.
    -------------
    Args:
        s0 ('torch.tensor'): [0, 1]
        s1 ('torch.tensor'): [-1, 1]
        s2 ('torch.tensor'): [-1, 1]
    Returns:
        i0 ('torch.tensor'): [0, 1]
        i45 ('torch.tensor'): [0, 1]
        i90 ('torch.tensor'): [0, 1]
        i135 ('torch.tensor'): [0, 1]
    """
    i0 = (s0 + s1)
    i45 = (s0 + s2)
    i90 = (s0 - s1)
    i135 = (s0 - s2)

    return i0, i45, i90, i135

def calc_dop_from_stokes(s0, s1, s2):
    """ 
    This function calculates the dop (degree of polarization)
    by using the input s0, s1, s2 images.
    -------------
    Args:
        s0 ('torch.tensor'): [0, 1]
        s1 ('torch.tensor'): [-1, 1]
        s2 ('torch.tensor'): [-1, 1]
    Returns:
        dop ('torch.tensor'): [0, 1]
    """
    s0_tmp = s0.clone()
    invalid = (s0_tmp == 0)
    s0_tmp[invalid] += 0.0001

    dop = torch.sqrt(s1 ** 2 + s2 ** 2) / s0_tmp
    dop[invalid] = 0
    dop = torch.clamp(dop, 0.0, 1.0)

    return dop


def calc_polar_phase_from_stokes(s1, s2, radian=True):
    """ 
    This function calculates the phase (polarization phase)
    by using the input s1 and s2 images.
    -------------
    Args:
        s1 ('torch.tensor'): [-1, 1]
        s2 ('torch.tensor'): [-1, 1]
    Returns:
        phase ('torch.tensor'): Polarization phase (degree), [0, 180).
    """
    invalid = (s1 == 0) & (s2 == 0)
    phase = rad2deg(torch.atan2(s2, s1))
    phase[phase < 0] = phase[phase < 0] + PHASEMAX_DEG * 2
    phase = phase / 2.
    phase[invalid] = 0

    phase = torch.clamp(phase, 0, PHASEMAX_DEG)

    if radian:
        phase = phase / 180. * np.pi

    return phase

def calc_admap_from_stokes(s0, s1, s2, forvis=False):
    """ 
    This function calculates the admap
    by using the input s0, s1, s2 images.
    -------------
    Args:
        s0 ('torch.tensor'): [0, 1]
        s1 ('torch.tensor'): [-1, 1]
        s2 ('torch.tensor'): [-1, 1]
    Returns:
        admap ('torch.tensor'): [0, 1]
    """
    _, _, admap = calc_dolp_aolp_admap_from_stokes(s0, s1, s2, forvis=forvis)
    return admap

def calc_dolp_aolp_admap_from_stokes(s0, s1, s2, forvis=False):
    """ 
    This function calculates the dolp, aolp, admap
    by using the input s0, s1, s2 images.
    -------------
    Args:
        s0 ('torch.tensor'): [0, 1]
        s1 ('torch.tensor'): [-1, 1]
        s2 ('torch.tensor'): [-1, 1]
    Returns:
        dolp ('torch.tensor'): [0, 1]
        aolp ('torch.tensor'): [0, pi).
        admap ('torch.tensor'): [-1, 1]
    """

    dolp = calc_dop_from_stokes(s0, s1, s2)
    aolp = calc_polar_phase_from_stokes(s1, s2)

    if forvis:
        dolp = dolp ** 0.5

    r = torch.cos(2.0*aolp)*dolp
    g = torch.sin(2.0*aolp)*dolp
    b = (1.0-dolp**2.0)**0.5

    admap = torch.stack([r,g,b], dim=1)

    return dolp, aolp, admap

def calc_diffuse_dop_from_zenith_deg(zenith_deg, ref_id):
    """
    This function calculates dop (degree of polarization) from input zenith angle and refractive index,
    based on the diffuse polarization model.
    -------------
    Args:
        zenith_deg ('torch.tensor'): Zenith angle map (degree), [0, 90].
        ref_id ('float'): Refractive index.
    Returns:
        difs_dop ('torch.tensor'): [0, 1]
    """
    sin_val = torch.sin(torch.deg2rad(zenith_deg))
    cos_val = torch.cos(torch.deg2rad(zenith_deg))

    difs_dop = (ref_id - 1 / ref_id) ** 2 * sin_val ** 2 / \
               (2 + 2 * ref_id ** 2 - (ref_id + 1 / ref_id) ** 2 * sin_val ** 2 + 4 * cos_val *
                torch.sqrt(ref_id ** 2 - sin_val ** 2))

    return difs_dop


def calc_diffuse_zenith_from_dop(in_dop, ref_id):
    """
    This function calculates dop (degree of polarization) from input zenith angle and refractive index,
    based on the diffuse polarization model.
    -------------
    Args:
        in_dop ('torch.tensor'): Degree of polarization, [0, 1].
        ref_id ('float'): Refractive index.
    Returns:
        out_zenith ('torch.tensor'): Zenith angle map (degree), [0, 90].
    """
    max_dop = calc_diffuse_dop_from_zenith_deg(torch.tensor(ZENITHMAX_DEG), ref_id)
    in_dop = torch.clamp(in_dop, 0, max_dop)

    val_A = 2 * (1 - in_dop) - (1 + in_dop) * (ref_id ** 2 + 1 / (ref_id ** 2))
    val_B = 4 * in_dop
    val_C = 1 + ref_id ** 2
    val_D = 1 - ref_id ** 2

    sin_val_sqrt_num = -1 * val_B * (val_C * (val_A + val_B) - torch.sqrt(
        val_C ** 2 * (val_A + val_B) ** 2 - val_D ** 2 * (val_A ** 2 - val_B ** 2)))
    sin_val_sqrt_denum = 2 * (val_A ** 2 - val_B ** 2)

    sin_val = sin_val_sqrt_num / sin_val_sqrt_denum
    sin_val = torch.clamp(sin_val, 0, 1)

    sin_val_mask = (sin_val == 0)
    sin_val[sin_val_mask] += 0.0001

    out_zenith = rad2deg(torch.arcsin(torch.sqrt(sin_val)))
    out_zenith[sin_val_mask] = 0
    out_zenith = torch.clamp(out_zenith, 0, ZENITHMAX_DEG)

    return out_zenith


def calc_normal_from_dop_and_phase(im_dop, im_phase, range01=False, radian_aolp=True):
    """
    This function calculates surface normal map using input dop and phase.
    -------------
    Args:
        im_dop ('torch.tensor'): Degree of polarization, [0, 1].
        im_phase ('torch.tensor'):  Polarization phase (degree), [0, 180).
    Returns:
        out_normal ('torch.tensor'): Surface normal map (h, w, xyz), [-1, 1].
    """
    if not radian_aolp:
        im_phase = torch.deg2rad(im_phase)
    zenith = calc_diffuse_zenith_from_dop(im_dop, REF_ID)

    normal_x = torch.cos(im_phase) * torch.sin(torch.deg2rad(zenith))
    normal_y = torch.sin(im_phase) * torch.sin(torch.deg2rad(zenith))
    normal_z = torch.cos(torch.deg2rad(zenith))

    out_normal = torch.stack([normal_x, normal_y, normal_z], dim=1)

    if range01:
        out_normal = (out_normal + 1) / 2
        out_normal = torch.clamp(out_normal, 0, 1)

    return out_normal


def calc_normal_from_stokes(s0, s1, s2):
    """ 
    This function calculates the admap
    by using the input s0, s1, s2 images.
    -------------
    Args:
        s0 ('torch.tensor'): [0, 1]
        s1 ('torch.tensor'): [-1, 1]
        s2 ('torch.tensor'): [-1, 1]
    Returns:
        normal ('torch.tensor'): [-1, 1]
    """
    dolp = calc_dop_from_stokes(s0, s1, s2)
    aolp = calc_polar_phase_from_stokes(s1, s2)
    normal = calc_normal_from_dop_and_phase(dolp, aolp, radian_aolp=True)
    return normal