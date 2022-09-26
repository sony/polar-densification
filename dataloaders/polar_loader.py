import os
import os.path
import glob
import fnmatch  # pattern matching
import numpy as np
from numpy import linalg as LA
from random import choice
from PIL import Image
import torch
import torch.utils.data as data
import cv2
from dataloaders import transforms

from util.util import MAX_8BIT, MAX_16BIT
from util.polarutils import getmask, mulmask

def get_globs(split, data_folder, gt_folder):

    globs = {}

    globs['s0'] = os.path.join(
        data_folder,
        split,
        's0/*s0.png'
    )
    globs['s0sps'] = os.path.join(
        data_folder,
        split,
        's0sps/*s0sps.png'
    )
    globs['s1sps'] = os.path.join(
        data_folder,
        split,
        's1sps/*s1sps.png'
    )
    globs['s2sps'] = os.path.join(
        data_folder,
        split,
        's2sps/*s2sps.png'
    )

    globs['s0gt'] = os.path.join(
        gt_folder,
        split,
        's0gt/*s0gt.png'
    )
    globs['s1gt'] = os.path.join(
        gt_folder,
        split,
        's1gt/*s1gt.png'
    )
    globs['s2gt'] = os.path.join(
        gt_folder,
        split,
        's2gt/*s2gt.png'
    )
    return globs

def get_paths_from_globs(globs):
    paths = {}
    paths['s0'] = sorted(glob.glob(globs['s0']))
    paths['s0sps'] = sorted(glob.glob(globs['s0sps']))
    paths['s1sps'] = sorted(glob.glob(globs['s1sps']))
    paths['s2sps'] = sorted(glob.glob(globs['s2sps']))
    paths['s0gt'] = sorted(glob.glob(globs['s0gt']))
    paths['s1gt'] = sorted(glob.glob(globs['s1gt']))
    paths['s2gt'] = sorted(glob.glob(globs['s2gt']))
    return paths

def add_paths_from_globs(paths, globs):
    paths['s0'] += sorted(glob.glob(globs['s0']))
    paths['s0sps'] += sorted(glob.glob(globs['s0sps']))
    paths['s1sps'] += sorted(glob.glob(globs['s1sps']))
    paths['s2sps'] += sorted(glob.glob(globs['s2sps']))
    paths['s0gt'] += sorted(glob.glob(globs['s0gt']))
    paths['s1gt'] += sorted(glob.glob(globs['s1gt']))
    paths['s2gt'] += sorted(glob.glob(globs['s2gt']))
    return paths


def print_paths_length(paths):
    print('s0:{}'.format(len(paths['s0'])))
    print('s0sps:{}'.format(len(paths['s0sps'])))
    print('s1sps:{}'.format(len(paths['s1sps'])))
    print('s2sps:{}'.format(len(paths['s2sps'])))
    print('s0gt:{}'.format(len(paths['s0gt'])))
    print('s1gt:{}'.format(len(paths['s1gt'])))
    print('s2gt:{}'.format(len(paths['s2gt'])))

def get_paths_and_transform(split, args):
    if split == 'train':
        transform = train_transform
    elif split == 'val':
        transform = val_transform
    elif split == 'test':
        transform = no_transform
    else:
        raise ValueError("Unrecognized split " + str(split))

    globs = get_globs(split, args.data_folder, args.gt_folder)
    paths = get_paths_from_globs(globs)

    if len(paths['s0']) == 0 or  len(paths['s0gt']) == 0 or len(paths['s1gt']) == 0 or len(paths['s2gt']) == 0:
        print_paths_length(paths)
        raise (RuntimeError("Found 0 images."))

    if not (len(paths['s0']) == len(paths['s0gt']) == len(paths['s1gt']) == len(paths['s2gt'])):
        print_paths_length(paths)
        raise (RuntimeError("The number of images is different."))

    if not args.raw_pattern=='quad':
        if not (len(paths['s0']) == len(paths['s0sps']) == len(paths['s1sps']) == len(paths['s2sps'])):
            print_paths_length(paths)
            raise (RuntimeError("The number of images is different."))

    return paths, transform


def s0_read(filename, gamma=False, s0_8bit=True):
    # Load 16bit image -> float [0:1]
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = cv2.imread(filename, -1)
    s0 = img_file.astype(dtype=np.float32) / MAX_16BIT # scale pixels to the range [0,1]
    if gamma:
        s0 = s0 ** (1.0/2.2)
    s0 = s0 * MAX_8BIT # Scale to 8bit [0, 255]
    if s0_8bit:
        s0 = s0.astype(dtype='uint8')
    return s0


def s12_read(filename):
    # Load 16bit image -> float [-1:1]
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = cv2.imread(filename, -1)
    s12 = img_file.astype(dtype=np.float32) / MAX_16BIT * 2. - 1. # scale pixels to the range [-1,1]
    s12 *= MAX_8BIT # Scale to 8bit [-255, 255]
    return s12


def crop(img, i, j, rh, rw):
    if img is not None:
        if img.ndim == 3:
            img = img[i:i + rh, j:j + rw, :]
        elif img.ndim == 2:
            img = img[i:i + rh, j:j + rw]
    return img


def train_transform(source, target, args):
    oheight = args.val_h
    owidth = args.val_w

    do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

    transforms_list = [
        transforms.BottomCrop((oheight, owidth)), # Delete the top part and crop to Output size
        transforms.HorizontalFlip(do_flip) # Flip horizontally at random (do_flip)
    ]

    transform_geometric = transforms.Compose(transforms_list)

    if source['s0sps'] is not None:
        source['s0sps'] = transform_geometric(source['s0sps'])
    if source['s1sps'] is not None:
        source['s1sps'] = transform_geometric(source['s1sps'])
    if source['s2sps'] is not None:
        source['s2sps'] = transform_geometric(source['s2sps'])
    if source['mask'] is not None:
        source['mask'] = transform_geometric(source['mask'])
        
    if target['s0gt'] is not None:
        target['s0gt'] = transform_geometric(target['s0gt'])
    if target['s1gt'] is not None:
        target['s1gt'] = transform_geometric(target['s1gt'])
    if target['s2gt'] is not None:
        target['s2gt'] = transform_geometric(target['s2gt'])

    if source['s0'] is not None:
        brightness = np.random.uniform(max(0, 1 - args.jitter),
                                       1 + args.jitter)
        contrast = np.random.uniform(max(0, 1 - args.jitter), 1 + args.jitter)
        saturation = np.random.uniform(max(0, 1 - args.jitter),
                                       1 + args.jitter)
        if args.s0_8bit:
            transform_s0 = transforms.Compose([
                transforms.ColorJitter(brightness, contrast, saturation, 0),
                transform_geometric
            ])
        else:
            transform_s0 = transform_geometric
        source['s0'] = transform_s0(source['s0'])


    # random crop
    if args.not_random_crop == False:
        h = oheight
        w = owidth
        rheight = args.random_crop_height
        rwidth = args.random_crop_width
        # randomlize
        i = np.random.randint(0, h - rheight + 1)
        j = np.random.randint(0, w - rwidth + 1)

        if source['s0'] is not None:
            source['s0'] = crop(source['s0'], i, j, rheight, rwidth)
        if source['s0sps'] is not None:
            source['s0sps'] = crop(source['s0sps'], i, j, rheight, rwidth)
        if source['s1sps'] is not None:
            source['s1sps'] = crop(source['s1sps'], i, j, rheight, rwidth)
        if source['s2sps'] is not None:
            source['s2sps'] = crop(source['s2sps'], i, j, rheight, rwidth)
        if target['s0gt'] is not None:
            target['s0gt'] = crop(target['s0gt'], i, j, rheight, rwidth)
        if target['s1gt'] is not None:
            target['s1gt'] = crop(target['s1gt'], i, j, rheight, rwidth)
        if target['s2gt'] is not None:
            target['s2gt'] = crop(target['s2gt'], i, j, rheight, rwidth)
        if source['mask'] is not None:
            source['mask'] = crop(source['mask'], i, j, rheight, rwidth)

    return source, target

def val_transform(source, target, args):
    oheight = args.val_h
    owidth = args.val_w

    # Only Bottom crop
    transform = transforms.Compose([
        transforms.BottomCrop((oheight, owidth)),
    ])

    if source['s0'] is not None:
        source['s0'] = transform(source['s0'])
    if source['s0sps'] is not None:
        source['s0sps'] = transform(source['s0sps'])
    if source['s1sps'] is not None:
        source['s1sps'] = transform(source['s1sps'])
    if source['s2sps'] is not None:
        source['s2sps'] = transform(source['s2sps'])
    if target['s0gt'] is not None:
        target['s0gt'] = transform(target['s0gt'])
    if target['s1gt'] is not None:
        target['s1gt'] = transform(target['s1gt'])
    if target['s2gt'] is not None:
        target['s2gt'] = transform(target['s2gt'])
    if source['mask'] is not None:
        source['mask'] = transform(source['mask'])

    return source, target

def no_transform(source, target, args):
    return source, target


to_tensor = transforms.ToTensor()
to_float_tensor = lambda x: to_tensor(x).float()


def handle_color(img, gray=True):
    if img is None:
        return None

    if img.ndim == 2:
        img = np.expand_dims(img, -1)

    if gray:
        if img.shape[2] == 3:
            gray = (img[:,:,0] + img[:,:,1] + img[:,:,2])/3.0
            return gray
        else:
            return img
    else:
        if img.shape[2] == 1:
            rgb = np.concatenate([img, img, img], axis=2)
            return rgb
        else:
            return img

class PolarCG(data.Dataset):
    """A data loader for the Polarization dataset
    """

    def __init__(self, split, args):
        self.args = args
        self.split = split
        paths, transform = get_paths_and_transform(split, args)
        self.paths = paths
        self.transform = transform
        self.threshold_translation = 0.1
        self.rp = args.raw_pattern

    def __getraw__(self, index):
        source = {}
        target = {}

        source['s0'] = s0_read(self.paths['s0'][index], s0_8bit=self.args.s0_8bit) if \
            (self.paths['s0'][index] is not None) else None

        source['s0sps'] = s0_read(self.paths['s0sps'][index], s0_8bit=self.args.s0_8bit) if \
            (self.paths['s0sps'][index] is not None) else None
        source['s1sps'] = s12_read(self.paths['s1sps'][index]) if \
            (self.paths['s1sps'][index] is not None) else None
        source['s2sps'] = s12_read(self.paths['s2sps'][index]) if \
            (self.paths['s2sps'][index] is not None) else None


        target['s0gt'] = s0_read(self.paths['s0gt'][index], s0_8bit=self.args.s0_8bit) if self.paths['s0gt'][index] is not None else None
        target['s1gt'] = s12_read(self.paths['s1gt'][index]) if self.paths['s1gt'][index] is not None else None
        target['s2gt'] = s12_read(self.paths['s2gt'][index]) if self.paths['s2gt'][index] is not None else None

        return source, target

    def __getitem__(self, index):
        # Get images from paths
        source, target = self.__getraw__(index)

        # Mul and get mask
        source['s0sps'] = mulmask(source['s0sps'], self.rp, polar_zero=False)
        source['s1sps'] = mulmask(source['s1sps'], self.rp, polar_zero=False)
        source['s2sps'] = mulmask(source['s2sps'], self.rp, polar_zero=False)
        source['mask'] = getmask(source['s0'].shape[:2], self.rp)[..., None] * 255.0

        source, target = self.transform(source, target, self.args)

        source['s0sps'] = handle_color(source['s0sps'], gray=True)
        source['s1sps'] = handle_color(source['s1sps'], gray=True)
        source['s2sps'] = handle_color(source['s2sps'], gray=True)
        target['s1gt'] = handle_color(target['s1gt'], gray=True)
        target['s2gt'] = handle_color(target['s2gt'], gray=True)

        candidates = {'mask': source['mask'], 's0': source['s0'], \
                      's0sps': source['s0sps'], 's1sps': source['s1sps'], 's2sps': source['s2sps'], \
                      's0gt': target['s0gt'], 's1gt': target['s1gt'], 's2gt': target['s2gt']}

        items = {
            key: to_float_tensor(val)
            for key, val in candidates.items() if val is not None
        }

        return items

    # Get dataset size
    def __len__(self):
        return len(self.paths['s0gt'])
