"""
These functions deal with polarization information.
"""
import numpy as np

def getmask(shape, raw_pattern='x64', offset=0.0, nomask=False):
    if raw_pattern=='x64':
        dst = getmask_x64(shape, offset, nomask)
    elif raw_pattern=='x16':
        dst = getmask_x16(shape, offset, nomask)
    elif raw_pattern=='x4':
        dst = getmask_x4(shape, offset, nomask)
    elif raw_pattern=='conv':
        dst = getmask_conv(shape, offset, nomask)
    elif raw_pattern=='quad':
        dst = getmask_quad(shape, offset, nomask)
    else:
        print('Unexpected Raw Pattern:', raw_pattern)
        exit()
    return dst
    
def getmask_quad(shape, offset=0.0, nomask=False):
    """
    This function return mask of binnig polar array.
    -------------
    Returns:
        mask ('numpy.ndarray'): Mask image [0, 1].
    """
    mask = np.zeros(shape) + offset
    return mask

def getmask_conv(shape, offset=0.0, nomask=False):
    """
    This function return mask of binnig polar array.
    -------------
    Returns:
        mask ('numpy.ndarray'): Mask image [0, 1].
    """
    mask = np.zeros(shape) + offset
    if not nomask:
        mask[:, :] = 1.0
    return mask

def getmask_x4(shape, offset=0.0, nomask=False):
    """
    This function return mask of binnig polar array.
    -------------
    Returns:
        mask ('numpy.ndarray'): Mask image [0, 1].
    """
    mask = np.zeros(shape) + offset
    ph, pw = 2, 0
    if not nomask:
        mask[ph::4, pw::4] = 1.0
        mask[ph+1::4, pw::4] = 1.0
        mask[ph::4, pw+1::4] = 1.0
        mask[ph+1::4, pw+1::4] = 1.0
    return mask

def getmask_x16(shape, offset=0.0, nomask=False):
    """
    This function return mask of binnig polar array.
    -------------
    Returns:
        mask ('numpy.ndarray'): Mask image [0, 1].
    """
    mask = np.zeros(shape) + offset
    ph, pw = 2, 2
    if not nomask:
        mask[ph::8, pw::8] = 1.0
        mask[ph+1::8, pw::8] = 1.0
        mask[ph::8, pw+1::8] = 1.0
        mask[ph+1::8, pw+1::8] = 1.0
    return mask

def getmask_x64(shape, offset=0.0, nomask=False):
    """
    This function return mask of binnig polar array.
    -------------
    Returns:
        mask ('numpy.ndarray'): Mask image [0, 1].
    """
    mask = np.zeros(shape) + offset
    ph, pw = 6, 6
    if not nomask:
        mask[ph::16, pw::16] = 1.0
        mask[ph+1::16, pw::16] = 1.0
        mask[ph::16, pw+1::16] = 1.0
        mask[ph+1::16, pw+1::16] = 1.0
    return mask

def mulmask(src, raw_pattern='x64', polar_zero=False):
    if raw_pattern=='x64':
        dst = mulmask_x64(src, polar_zero)
    elif raw_pattern=='x16':
        dst = mulmask_x16(src, polar_zero)
    elif raw_pattern=='x4':
        dst = mulmask_x4(src, polar_zero)
    elif raw_pattern=='conv':
        dst = mulmask_conv(src, polar_zero)
    elif raw_pattern=='quad':
        dst = mulmask_quad(src, polar_zero)
    else:
        print('Unexpected Raw Pattern:', raw_pattern)
        exit()
    return dst

def mulmask_quad(img, polar_zero=False):
    """
    This function return mask of binnig polar array.
    -------------
    Returns:
        mask ('numpy.ndarray'): Mul Mask image.
    """
    if polar_zero:    
        mask = np.ones_like(img)
    else:
        mask = np.zeros_like(img)

    return img * mask


def mulmask_conv(img, polar_zero=False):
    """
    This function return mask of binnig polar array.
    -------------
    Returns:
        mask ('numpy.ndarray'): Mul Mask image.
    """
    if polar_zero:    
        mask = np.zeros_like(img)
    else:
        mask = np.ones_like(img)

    return img * mask

def mulmask_x4(img, polar_zero=False):
    """
    This function return mask of binnig polar array.
    -------------
    Returns:
        mask ('numpy.ndarray'): Mul Mask image.
    """
    ph, pw = 2, 0

    if polar_zero:    
        mask = np.ones_like(img)
        mask[ph::4, pw::4] = 0.0
        mask[ph+1::4, pw::4] = 0.0
        mask[ph::4, pw+1::4] = 0.0
        mask[ph+1::4, pw+1::4] = 0.0
    else:
        mask = np.zeros_like(img)
        mask[ph::4, pw::4] = 1.0
        mask[ph+1::4, pw::4] = 1.0
        mask[ph::4, pw+1::4] = 1.0
        mask[ph+1::4, pw+1::4] = 1.0

    return img * mask

def mulmask_x16(img, polar_zero=False):
    """
    This function return mask of binnig polar array.
    -------------
    Returns:
        mask ('numpy.ndarray'): Mul Mask image.
    """
    ph, pw = 2, 2

    if polar_zero:    
        mask = np.ones_like(img)
        mask[ph::8, pw::8] = 0.0
        mask[ph+1::8, pw::8] = 0.0
        mask[ph::8, pw+1::8] = 0.0
        mask[ph+1::8, pw+1::8] = 0.0
    else:
        mask = np.zeros_like(img)
        mask[ph::8, pw::8] = 1.0
        mask[ph+1::8, pw::8] = 1.0
        mask[ph::8, pw+1::8] = 1.0
        mask[ph+1::8, pw+1::8] = 1.0

    return img * mask


def mulmask_x64(img, polar_zero=False):
    """
    This function return mask of binnig polar array.
    -------------
    Returns:
        mask ('numpy.ndarray'): Mul Mask image.
    """
    ph, pw = 6, 6

    if polar_zero:    
        mask = np.ones_like(img)
        mask[ph::16, pw::16] = 0.0
        mask[ph+1::16, pw::16] = 0.0
        mask[ph::16, pw+1::16] = 0.0
        mask[ph+1::16, pw+1::16] = 0.0
    else:
        mask = np.zeros_like(img)
        mask[ph::16, pw::16] = 1.0
        mask[ph+1::16, pw::16] = 1.0
        mask[ph::16, pw+1::16] = 1.0
        mask[ph+1::16, pw+1::16] = 1.0

    return img * mask