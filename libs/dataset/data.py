import torch
import os
import math
import cv2
import numpy as np

import json
import yaml
import random
import pickle

from PIL import Image
from torch.utils.data import Dataset

from ..utils.logger import getLogger

__DATA_CONTAINER = {}

ROOT = '/home/ubuntu11/zzc/data'
MAX_TRAINING_OBJ = 5
MAX_TRAINING_SKIP = 120

class DistributedLoader(object):

    def __init__(self, dataset, sampler, collate_fn):

        self.dataset = dataset
        self.sampler = sampler
        self.collate_fn = collate_fn

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
 
        for idx in self.sampler:
            yield self.collate_fn([self.dataset[idx]])

def register_data(name, dataset):
    if name in __DATA_CONTAINER:
        raise TypeError('dataset with name {} has already been registered'.format(name))

    __DATA_CONTAINER[name] = dataset
    dataset.set_alias(name)

def build_dataset(name, *args, **kwargs):

    logger = getLogger(__name__)

    if name not in __DATA_CONTAINER:
        logger.error('invalid dataset name is encountered. The current acceptable datasets are:')
        support_sets = ' '.join(list(__DATA_CONTAINER.keys()))
        logger.error(support_sets)
        raise TypeError('name not found for dataset {}'.format(name))
    # print(name)
    return __DATA_CONTAINER[name](*args, **kwargs)

def multibatch_collate_fn(batch):

    min_time = min([sample[0].shape[0] for sample in batch])
    frames = torch.stack([sample[0] for sample in batch])
    masks = torch.stack([sample[1] for sample in batch])
    coord = torch.stack([sample[2] for sample in batch])
    corner = torch.stack([sample[3] for sample in batch])
    objs = [sample[4] for sample in batch]

    try:
        info = [sample[5] for sample in batch]
    except IndexError as ie:
        info = None

    return frames, masks, coord, corner, objs, info


def test_multibatch_collate_fn(batch):
    
    min_time = min([sample[0].shape[0] for sample in batch])
    frames = torch.stack([sample[0] for sample in batch])
    masks = torch.stack([sample[1] for sample in batch])
    coord = torch.stack([sample[2] for sample in batch])
    corner = torch.stack([sample[3] for sample in batch])
    objs = [sample[4] for sample in batch]

    try:
        info = [sample[5] for sample in batch]
    except IndexError as ie:
        info = None

    return frames, masks, coord, corner, objs, info

def multibatch_collate_fn_center(batch):
    
    min_time = min([sample[0].shape[0] for sample in batch])
    frames = torch.stack([sample[0] for sample in batch])
    masks = torch.stack([sample[1] for sample in batch])
    coord = torch.stack([sample[2] for sample in batch])
    corner = torch.stack([sample[3] for sample in batch])
    center = torch.stack([sample[4] for sample in batch])
    objs = [sample[5] for sample in batch]

    try:
        info = [sample[6] for sample in batch]
    except IndexError as ie:
        info = None

    return frames, masks, coord, corner, center, objs, info


def test_multibatch_collate_fn_center(batch):
    
    min_time = min([sample[0].shape[0] for sample in batch])
    frames = torch.stack([sample[0] for sample in batch])
    masks = torch.stack([sample[1] for sample in batch])
    coord = torch.stack([sample[2] for sample in batch])
    corner = torch.stack([sample[3] for sample in batch])
    center = torch.stack([sample[4] for sample in batch])
    objs = [sample[5] for sample in batch]

    try:
        info = [sample[6] for sample in batch]
    except IndexError as ie:
        info = None

    return frames, masks, coord, corner, center, objs, info


def convert_mask(mask, max_obj):

    # convert mask to one hot encoded
    oh = []
    for k in range(max_obj+1):
        oh.append(mask==k)

    if isinstance(mask, np.ndarray):
        oh = np.stack(oh, axis=-1)
    else:
        oh = torch.cat(oh, dim=-1).float()

    return oh

def convert_one_hot(oh, max_obj):

    if isinstance(oh, np.ndarray):
        mask = np.zeros(oh.shape[:2], dtype=np.uint8)
    else:
        mask = torch.zeros(oh.shape[:2])

    for k in range(max_obj+1):
        mask[oh[:, :, k]==1] = k

    return mask

class BaseData(Dataset):

    alias = None

    def increase_max_skip(self):
        pass

    def set_max_skip(self):
        pass

    @classmethod
    def set_alias(cls, name):
        cls.alias = name

    @classmethod
    def get_alias(cls):
        return cls.alias


