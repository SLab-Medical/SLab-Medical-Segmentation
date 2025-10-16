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

class liuxiangyue(torch.utils.data.Dataset):
    def __init__(self, args):

        self.subjects = []

        images_dir = Path(os.path.join(args.dataset_path, 'train'))
        image_paths = sorted(images_dir.glob("*/CT_Image.mha"))[:10]
        labels_dir = Path(os.path.join(args.dataset_path, 'train'))
        label_paths = sorted(labels_dir.glob("*/Heart.mha"))[:10]

        for (image_path, label_path) in zip(image_paths, label_paths):
            subject = tio.Subject(
                source=tio.ScalarImage(image_path),
                label=tio.LabelMap(label_path),
            )
            self.subjects.append(subject)

        self.transforms = self.transform()

        self.training_set = tio.SubjectsDataset(self.subjects, transform=self.transforms)

        self.queue_dataset = Queue(
            subjects_dataset=self.training_set,
            sampler=UniformSampler(args.patch_size),
            max_length=args.queue_length,
            samples_per_volume=args.samples_per_volume,
        )




    def transform(self):

        training_transform = Compose([
            # tio.ToCanonical(),  # to RAS
            # tio.Resample((1, 1, 1)),  # to 1 mm iso

            RescaleIntensity(out_min_max=(0, 1)),
            CropOrPad((128, 128, 128)),
            # ZNormalization(),
        ])



        return training_transform