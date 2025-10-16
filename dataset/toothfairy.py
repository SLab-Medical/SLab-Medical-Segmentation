from glob import glob
from os.path import dirname, join, basename, isfile
import sys
sys.path.append('./')
import csv
import torch
import numpy as np
from PIL import Image
from torch import nn
import torch.nn.functional as F
import random
import torchio as tio
from torchio import AFFINE, DATA
import torchio
from torchio import ScalarImage, LabelMap, Subject, SubjectsDataset, Queue
from torchio.data import UniformSampler
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
)
from pathlib import Path
import os

class ToothFairy(torch.utils.data.Dataset):
    def __init__(self, args):


        queue_length = args.queue_length
        samples_per_volume = args.samples_per_volume


        self.subjects = []


        images_dir = Path(os.path.join(args.data_root, 'img'))
        image_paths = sorted(images_dir.glob("*.nii.gz"))
        labels_dir = Path(os.path.join(args.data_root, 'label'))
        label_paths = sorted(labels_dir.glob("*.nii.gz"))

        for (image_path, label_path) in zip(image_paths, label_paths):
            subject = tio.Subject(
                source=tio.ScalarImage(image_path),
                label=tio.LabelMap(label_path),
            )
            self.subjects.append(subject)
       


        self.transforms = self.transform()

        self.training_set = tio.SubjectsDataset(self.subjects, transform=self.transforms)


        self.queue_dataset = Queue(
            self.training_set,
            queue_length,
            samples_per_volume,
            UniformSampler(args.patch_size),
            num_workers=0
        )




    def transform(self):

        training_transform = Compose([
            # tio.ToCanonical(),  # to RAS
            # tio.Resample((1, 1, 1)),  # to 1 mm iso

            RescaleIntensity(out_min_max=(0, 1)),
            # ZNormalization(),
        ])



        return training_transform