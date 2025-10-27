from glob import glob
from os.path import dirname, join, basename, isfile
import sys

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


class Brat(torch.utils.data.Dataset):
    def __init__(self, args):


        self.subjects = []

        for case_dir in sorted(Path(args.dataset_path).glob("BraTS20_Training_*")):
            if not case_dir.is_dir():
                continue

            subject = tio.Subject(
                source=tio.ScalarImage([
                    case_dir / f"{case_dir.name}_flair.nii.gz",
                    case_dir / f"{case_dir.name}_t1.nii.gz",
                    case_dir / f"{case_dir.name}_t1ce.nii.gz",
                    case_dir / f"{case_dir.name}_t2.nii.gz",
                ]),
                label=tio.LabelMap(case_dir / f"{case_dir.name}_seg.nii.gz"),
            )
            self.subjects.append(subject)

        self.transforms = self.transform()

        self.training_set = tio.SubjectsDataset(self.subjects, transform=self.transforms)

        self.queue_dataset = Queue(
            subjects_dataset=self.training_set,
            max_length=args.queue_length,
            sampler=UniformSampler(args.patch_size),
            samples_per_volume=args.samples_per_volume,
        )

    def transform(self):

        training_transform = Compose([
            # tio.ToCanonical(),  # to RAS
            # tio.Resample((1, 1, 1)),  # to 1 mm iso

            RescaleIntensity(out_min_max=(0, 1)),
            # ZNormalization(),
        ])

        return training_transform