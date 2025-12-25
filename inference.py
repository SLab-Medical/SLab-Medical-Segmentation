"""
Inference script for SMS Medical Segmentation Toolkit
Supports both 2D and 3D segmentation with automatic detection
Calculates comprehensive metrics and saves results
"""

import os
import argparse
import time
from pathlib import Path
import yaml

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torchio as tio
from PIL import Image

from config.options import TrainOptions
from models.model_utils import get_model
from utils.metrics import calculate_metrics
from dataset.utils import build_dataset


class InferenceConfig:
    """Configuration for inference"""
    def __init__(self, args):
        self.checkpoint_path = args.checkpoint_path
        self.input_path = args.input_path
        self.output_dir = args.output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = args.batch_size
        self.save_probability = args.save_probability
        self.calculate_metrics = args.calculate_metrics
        self.use_sliding_window = args.use_sliding_window
        self.overlap = args.overlap


def load_checkpoint(checkpoint_path, model, device):
    """
    Load model checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance
        device: Device to load to

    Returns:
        Loaded model and training info
    """
    print(f"Loading checkpoint from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'model' in checkpoint:
        model_state_dict = checkpoint['model']
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"Checkpoint from epoch: {epoch}")
    elif 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 'unknown')
    else:
        model_state_dict = checkpoint
        epoch = 'unknown'

    # Load state dict
    try:
        model.load_state_dict(model_state_dict)
    except RuntimeError as e:
        print(f"Warning: {e}")
        print("Attempting to load with strict=False")
        model.load_state_dict(model_state_dict, strict=False)

    model.to(device)
    model.eval()

    print("Checkpoint loaded successfully!")
    return model, epoch


def inference_2d_single_image(model, image_path, device, args):
    """
    Perform inference on a single 2D image

    Args:
        model: Trained model
        image_path: Path to input image
        device: Device to run on
        args: Configuration arguments

    Returns:
        prediction, probability (if requested), and metrics (if ground truth available)
    """
    # Load image
    if image_path.suffix in ['.png', '.jpg', '.jpeg', '.bmp']:
        # Load as PIL image
        img = Image.open(image_path).convert('L')
        img_array = np.array(img, dtype=np.float32)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    elif image_path.suffix in ['.nii', '.gz', '.nii.gz']:
        # Load as NIfTI
        img = tio.ScalarImage(image_path)
        img_tensor = img.data  # [1, H, W, D]
        if img_tensor.shape[-1] == 1:
            img_tensor = img_tensor.squeeze(-1)  # [1, H, W]
        img_tensor = img_tensor.unsqueeze(0)  # [1, 1, H, W]
    else:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")

    # Normalize
    img_tensor = img_tensor / img_tensor.max() if img_tensor.max() > 0 else img_tensor
    img_tensor = img_tensor.to(device)

    # Inference
    with torch.no_grad():
        output = model(img_tensor)

    # Get prediction
    if args.model.out_channels == 1:
        # Binary segmentation
        prob = torch.sigmoid(output)
        pred = (prob > 0.5).float()
    else:
        # Multi-class segmentation
        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1, keepdim=True).float()

    return pred, prob, img_tensor


def inference_3d_sliding_window(model, image_path, device, args, patch_size, overlap):
    """
    Perform inference on a 3D volume using sliding window approach

    Args:
        model: Trained model
        image_path: Path to input volume
        device: Device to run on
        args: Configuration arguments
        patch_size: Size of sliding window patches
        overlap: Overlap between patches

    Returns:
        prediction, probability (if requested)
    """
    # Load 3D volume
    subject = tio.Subject(source=tio.ScalarImage(image_path))

    # Apply normalization
    znorm = tio.ZNormalization()
    subject = znorm(subject)

    # Create grid sampler
    grid_sampler = tio.inference.GridSampler(
        subject,
        patch_size,
        overlap,
    )

    # Create data loader
    patch_loader = torch.utils.data.DataLoader(
        grid_sampler,
        batch_size=1,
        num_workers=0
    )

    # Create aggregators
    aggregator_pred = tio.inference.GridAggregator(grid_sampler)
    aggregator_prob = tio.inference.GridAggregator(grid_sampler) if args.save_probability else None

    print(f"Processing {len(patch_loader)} patches...")

    # Process patches
    with torch.no_grad():
        for patches_batch in tqdm(patch_loader, desc="Inference"):
            input_tensor = patches_batch['source'][tio.DATA].to(device)
            locations = patches_batch[tio.LOCATION]

            # Forward pass
            output = model(input_tensor)

            # Get predictions
            if args.model.out_channels == 1:
                # Binary segmentation
                prob = torch.sigmoid(output)
                pred = (prob > 0.5).float()
            else:
                # Multi-class segmentation
                prob = torch.softmax(output, dim=1)
                pred = torch.argmax(output, dim=1, keepdim=True).float()

            # Add to aggregators
            aggregator_pred.add_batch(pred, locations)
            if aggregator_prob is not None:
                aggregator_prob.add_batch(prob, locations)

    # Get final outputs
    pred_tensor = aggregator_pred.get_output_tensor()
    prob_tensor = aggregator_prob.get_output_tensor() if aggregator_prob is not None else None

    return pred_tensor, prob_tensor, subject


def inference_3d_whole_volume(model, image_path, device, args):
    """
    Perform inference on a whole 3D volume (for small volumes)

    Args:
        model: Trained model
        image_path: Path to input volume
        device: Device to run on
        args: Configuration arguments

    Returns:
        prediction, probability (if requested)
    """
    # Load 3D volume
    img = tio.ScalarImage(image_path)
    img_tensor = img.data.unsqueeze(0)  # [1, C, D, H, W]

    # Normalize
    img_tensor = (img_tensor - img_tensor.mean()) / (img_tensor.std() + 1e-8)
    img_tensor = img_tensor.to(device)

    # Inference
    with torch.no_grad():
        output = model(img_tensor)

    # Get prediction
    if args.model.out_channels == 1:
        # Binary segmentation
        prob = torch.sigmoid(output)
        pred = (prob > 0.5).float()
    else:
        # Multi-class segmentation
        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1, keepdim=True).float()

    return pred, prob, img


def save_results_2d(pred, prob, output_path, save_probability=False):
    """Save 2D segmentation results"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save prediction as PNG
    pred_np = pred.squeeze().cpu().numpy()
    pred_img = Image.fromarray((pred_np * 255).astype(np.uint8))
    pred_img.save(output_path.with_suffix('.png'))

    # Save probability map if requested
    if save_probability and prob is not None:
        if prob.shape[1] == 1:
            # Binary
            prob_np = prob.squeeze().cpu().numpy()
            prob_img = Image.fromarray((prob_np * 255).astype(np.uint8))
            prob_img.save(output_path.with_name(output_path.stem + '_prob.png'))
        else:
            # Multi-class - save each class
            for c in range(prob.shape[1]):
                prob_np = prob[0, c].cpu().numpy()
                prob_img = Image.fromarray((prob_np * 255).astype(np.uint8))
                prob_img.save(output_path.with_name(f"{output_path.stem}_prob_class{c}.png"))


def save_results_3d(pred, prob, output_path, affine, save_probability=False):
    """Save 3D segmentation results"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save prediction as NIfTI
    pred_img = tio.ScalarImage(tensor=pred.cpu(), affine=affine)
    pred_img.save(output_path.with_suffix('.nii.gz'))

    # Save probability map if requested
    if save_probability and prob is not None:
        if prob.shape[0] == 1:
            # Binary
            prob_img = tio.ScalarImage(tensor=prob.cpu(), affine=affine)
            prob_img.save(output_path.with_name(output_path.stem + '_prob.nii.gz'))
        else:
            # Multi-class - save each class
            for c in range(prob.shape[0]):
                prob_img = tio.ScalarImage(tensor=prob[c:c+1].cpu(), affine=affine)
                prob_img.save(output_path.with_name(f"{output_path.stem}_prob_class{c}.nii.gz"))


def run_inference(args, config):
    """
    Main inference function

    Args:
        args: Training configuration
        config: Inference configuration
    """
    print("\n" + "="*80)
    print("SMS Medical Segmentation - Inference")
    print("="*80 + "\n")

    # Setup device
    device = config.device
    print(f"Using device: {device}")

    # Detect dimension from model config
    is_2d = args.model.dimension == '2d'
    print(f"Model dimension: {'2D' if is_2d else '3D'}")

    # Build model
    print("\nBuilding model...")
    model = get_model(args)
    print(f"Model: {args.model.model_name}")
    print(f"Input channels: {args.model.in_channels}")
    print(f"Output channels: {args.model.out_channels}")

    # Load checkpoint
    model, epoch = load_checkpoint(config.checkpoint_path, model, device)

    # Prepare input paths
    input_path = Path(config.input_path)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of input files
    if input_path.is_file():
        input_files = [input_path]
    elif input_path.is_dir():
        if is_2d:
            # 2D images
            input_files = list(input_path.glob('*.png')) + \
                         list(input_path.glob('*.jpg')) + \
                         list(input_path.glob('*.jpeg'))
        else:
            # 3D volumes
            input_files = list(input_path.glob('*.nii.gz')) + \
                         list(input_path.glob('*.nii'))
    else:
        raise ValueError(f"Input path not found: {input_path}")

    print(f"\nFound {len(input_files)} file(s) to process")

    # Initialize metrics storage
    all_metrics = []

    # Process each file
    print("\nStarting inference...\n")
    start_time = time.time()

    for idx, img_path in enumerate(input_files):
        print(f"Processing [{idx+1}/{len(input_files)}]: {img_path.name}")

        try:
            if is_2d:
                # 2D inference
                pred, prob, img_tensor = inference_2d_single_image(
                    model, img_path, device, args
                )

                # Save results
                output_path = output_dir / f"{img_path.stem}_pred"
                save_results_2d(pred, prob, output_path, config.save_probability)

                print(f"  ✓ Saved: {output_path}.png")

            else:
                # 3D inference
                if config.use_sliding_window:
                    # Use sliding window for large volumes
                    patch_size = getattr(args.dataset, 'patch_size', [96, 96, 96])
                    pred, prob, subject = inference_3d_sliding_window(
                        model, img_path, device, args,
                        patch_size, config.overlap
                    )
                    affine = subject['source']['affine']
                else:
                    # Process whole volume
                    pred, prob, img = inference_3d_whole_volume(
                        model, img_path, device, args
                    )
                    pred = pred.squeeze(0)  # Remove batch dimension
                    prob = prob.squeeze(0) if prob is not None else None
                    affine = img.affine

                # Save results
                output_path = output_dir / f"{img_path.stem}_pred"
                save_results_3d(pred, prob, output_path, affine, config.save_probability)

                print(f"  ✓ Saved: {output_path}.nii.gz")

            # Calculate metrics if ground truth is available
            if config.calculate_metrics:
                # Try to find ground truth file
                gt_path = img_path.parent.parent / 'label' / img_path.name
                if not gt_path.exists():
                    gt_path = img_path.parent / 'label' / img_path.name

                if gt_path.exists():
                    print(f"  Computing metrics with ground truth: {gt_path.name}")

                    # Load ground truth
                    if is_2d:
                        gt_img = Image.open(gt_path).convert('L')
                        gt_tensor = torch.from_numpy(np.array(gt_img, dtype=np.float32))
                        gt_tensor = gt_tensor.unsqueeze(0).unsqueeze(0).to(device)
                        gt_tensor = gt_tensor / 255.0
                    else:
                        gt_img = tio.ScalarImage(gt_path)
                        gt_tensor = gt_img.data.to(device)
                        if gt_tensor.dim() == 4:
                            gt_tensor = gt_tensor.unsqueeze(0)

                    # Compute metrics
                    with torch.no_grad():
                        pred_for_metric = pred.unsqueeze(0) if pred.dim() == 3 or pred.dim() == 4 else pred
                        metrics = calculate_metrics(
                            pred_for_metric,
                            gt_tensor,
                            num_classes=args.model.out_channels,
                            threshold=0.5,
                            include_distance_metrics=False  # Distance metrics are slow
                        )

                    # Print metrics
                    print(f"  Metrics:")
                    print(f"    Dice: {metrics['dice']:.4f}")
                    print(f"    IoU: {metrics['iou']:.4f}")
                    print(f"    Precision: {metrics['precision']:.4f}")
                    print(f"    Recall: {metrics['recall']:.4f}")
                    print(f"    F1: {metrics['f1']:.4f}")

                    metrics['filename'] = img_path.name
                    all_metrics.append(metrics)

        except Exception as e:
            print(f"  ✗ Error processing {img_path.name}: {str(e)}")
            continue

    # Calculate average metrics
    if all_metrics:
        print("\n" + "="*80)
        print("Average Metrics Across All Images")
        print("="*80)

        avg_metrics = {}
        for key in all_metrics[0].keys():
            if key != 'filename':
                avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        print(f"Dice:        {avg_metrics['dice']:.4f}")
        print(f"IoU:         {avg_metrics['iou']:.4f}")
        print(f"Precision:   {avg_metrics['precision']:.4f}")
        print(f"Recall:      {avg_metrics['recall']:.4f}")
        print(f"Specificity: {avg_metrics['specificity']:.4f}")
        print(f"F1 Score:    {avg_metrics['f1']:.4f}")
        print(f"Accuracy:    {avg_metrics['accuracy']:.4f}")

        # Save metrics to CSV
        import csv
        metrics_file = output_dir / 'metrics.csv'
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            writer.writerows(all_metrics)

        print(f"\nMetrics saved to: {metrics_file}")

    elapsed_time = time.time() - start_time
    print("\n" + "="*80)
    print(f"Inference completed in {elapsed_time:.2f} seconds")
    print(f"Average time per image: {elapsed_time/len(input_files):.2f} seconds")
    print(f"Results saved to: {output_dir}")
    print("="*80 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='SMS Medical Segmentation Inference')

    # Required arguments
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--input_path', type=str, required=True,
                       help='Path to input image/volume or directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save predictions')

    # Model configuration (must match training config)
    parser.add_argument('--config', type=str, default=None,
                       help='Path to training config YAML (optional, for loading exact config)')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Model name (e.g., unet_2d, segmamba)')
    parser.add_argument('--dimension', type=str, choices=['2d', '3d'], default=None,
                       help='Model dimension (2d or 3d)')
    parser.add_argument('--in_channels', type=int, default=None,
                       help='Number of input channels')
    parser.add_argument('--out_channels', type=int, default=None,
                       help='Number of output channels')

    # Inference options
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for inference')
    parser.add_argument('--use_sliding_window', action='store_true',
                       help='Use sliding window for 3D inference (recommended for large volumes)')
    parser.add_argument('--overlap', type=float, default=0.25,
                       help='Overlap ratio for sliding window')
    parser.add_argument('--save_probability', action='store_true',
                       help='Save probability maps in addition to predictions')
    parser.add_argument('--calculate_metrics', action='store_true',
                       help='Calculate metrics if ground truth is available')

    args_inference = parser.parse_args()

    # Load training configuration
    if args_inference.config:
        # Load from config file
        print(f"Loading configuration from: {args_inference.config}")
        args = TrainOptions().parse_from_yaml(args_inference.config)
    else:
        # Create config from command line arguments
        if not all([args_inference.model_name, args_inference.dimension,
                   args_inference.in_channels, args_inference.out_channels]):
            parser.error("Either --config or all of (--model_name, --dimension, "
                        "--in_channels, --out_channels) must be provided")

        # Create minimal config
        args = TrainOptions().parse([])
        args.model.model_name = args_inference.model_name
        args.model.dimension = args_inference.dimension
        args.model.in_channels = args_inference.in_channels
        args.model.out_channels = args_inference.out_channels

    # Create inference config
    config = InferenceConfig(args_inference)

    # Run inference
    run_inference(args, config)


if __name__ == '__main__':
    main()
