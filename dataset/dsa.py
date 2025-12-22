from glob import glob
from os.path import dirname, join, basename, isfile
import sys
import torch
import numpy as np
from PIL import Image
from torch import nn
import torch.nn.functional as F
import random
from pathlib import Path
import os
from torchvision import transforms
from torch.utils.data import Dataset


class DSA(Dataset):
    """
    2D DSA Dataset for medical image segmentation
    Data structure:
        - data/data_2d/DSA/training/img/*.png
        - data/data_2d/DSA/training/label/*.png
        - data/data_2d/DSA/test/img/*.png
        - data/data_2d/DSA/test/label/*.png
    """
    def __init__(self, args, mode='train'):
        """
        Args:
            args: dataset configuration arguments
            mode: 'train' or 'test'
        """
        self.args = args
        self.mode = mode

        # Set image size (use config if available, otherwise default to 512)
        self.image_size = getattr(args, 'image_size', 512)

        # Set paths
        if mode == 'train':
            data_dir = Path(args.dataset_path) / 'training'
        else:
            data_dir = Path(args.dataset_path) / 'test'

        img_dir = data_dir / 'img'
        label_dir = data_dir / 'label'

        # Get all image files
        self.image_paths = sorted(img_dir.glob("*.png"))
        self.label_paths = sorted(label_dir.glob("*.png"))

        # Ensure images and labels match
        assert len(self.image_paths) == len(self.label_paths), \
            f"Number of images ({len(self.image_paths)}) and labels ({len(self.label_paths)}) don't match"

        print(f"Loaded {len(self.image_paths)} {mode} samples from {data_dir}")

        # Setup transforms
        self.img_transform = self.get_img_transform()
        self.label_transform = self.get_label_transform()

    def get_img_transform(self):
        """Image transformations"""
        transform_list = [
            transforms.Resize((self.image_size, self.image_size)),  # Resize to consistent size
            transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
        ]

        if self.mode == 'train':
            # Add data augmentation for training
            pass  # Can add augmentation here if needed

        return transforms.Compose(transform_list)

    def get_label_transform(self):
        """Label transformations"""
        # Labels should be resized and converted to tensor
        # Use NEAREST interpolation for labels to preserve integer values
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and label
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # Read images
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        label = Image.open(label_path).convert('L')  # Convert to grayscale

        # Apply transforms
        img = self.img_transform(img)
        label = self.label_transform(label)

        # Convert label to binary if needed (threshold at 0.5)
        label = (label > 0.5).float()

        return {
            'source': {'data': img},
            'label': {'data': label}
        }


class DSAWithTorchio:
    """
    Wrapper class to provide TorchIO-like interface for compatibility with existing training code
    This is for 2D data but wrapped to match the 3D dataset interface
    """
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode

        # Create the base dataset
        self.dataset = DSA(args, mode)

        # For compatibility with existing code that expects queue_dataset
        self.queue_dataset = self.dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
