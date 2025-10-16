import builtins
import logging
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
# from model.utils import get_model
# from training.dataset.utils import get_dataset
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from config.options import TrainOptions
from copy import deepcopy

import yaml
import argparse
import time
import math
import sys
import pdb
import warnings
import datetime

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots

from utils.configuration import save_configure
from utils.resume import resume_load_model_checkpoint, resume_load_optimizer_checkpoint
from utils.ema import update_ema_variables, update_bn

from dataset.utils import build_dataset
from solver.optim import build_optimizer
from solver.scheduler import build_warmup_lr_scheduler, build_training_lr_scheduler
from models.model_utils import get_model


from losses import SegmentationLossManager

import types
import collections
from random import shuffle

from accelerate import Accelerator

import torchio as tio

warnings.filterwarnings("ignore", category=UserWarning)


def setup_logging(log_path, exp_name):
    """
    Setup comprehensive logging with file and console handlers
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(log_path, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # File handler for detailed logs
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'{exp_name}_detailed.log'),
        mode='w',
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler for important messages
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # Training specific logger
    train_logger = logging.getLogger('training')
    train_logger.setLevel(logging.INFO)
    
    # TorchIO specific logger
    torchio_logger = logging.getLogger('torchio')
    torchio_logger.setLevel(logging.INFO)
    
    # Model specific logger
    model_logger = logging.getLogger('model')
    model_logger.setLevel(logging.INFO)
    
    logging.info(f"Logging setup completed. Logs will be saved to: {log_dir}")
    return root_logger, train_logger, torchio_logger, model_logger


def setup_visualization_dirs(log_path):
    """
    Setup directories for saving visualizations and intermediate results
    """
    vis_dir = os.path.join(log_path, 'visualizations')
    intermediate_dir = os.path.join(log_path, 'intermediate_results')
    plots_dir = os.path.join(log_path, 'plots')
    
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(intermediate_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(vis_dir, 'input_images'), exist_ok=True)
    os.makedirs(os.path.join(vis_dir, 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(vis_dir, 'loss_curves'), exist_ok=True)
    os.makedirs(os.path.join(intermediate_dir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(intermediate_dir, 'predictions'), exist_ok=True)
    
    return vis_dir, intermediate_dir, plots_dir


def save_torchio_intermediate_results(inputs, predictions, epoch, iteration, intermediate_dir, torchio_logger):
    """
    Save intermediate results using TorchIO functionality
    """
    # Create a sample directory for this epoch and iteration
    sample_dir = os.path.join(intermediate_dir, 'samples', f'epoch_{epoch:04d}_iter_{iteration:04d}')
    os.makedirs(sample_dir, exist_ok=True)
    
    # Save input images and labels using TorchIO
    for batch_idx in range(inputs['source']['data'].shape[0]):
        # Create subject for this batch
        subject = tio.Subject(
            source=tio.ScalarImage(tensor=inputs['source']['data'][batch_idx].cpu()),
            label=tio.LabelMap(tensor=inputs['label']['data'][batch_idx].cpu()),
            prediction=tio.LabelMap(tensor=predictions[batch_idx].cpu())
        )
        
        subject.source.save(os.path.join(sample_dir, f"iter-{iteration:04d}-source.nii.gz"))
        subject.label.save(os.path.join(sample_dir, f"iter-{iteration:04d}-label.nii.gz"))
        subject.prediction.save(os.path.join(sample_dir, f"iter-{iteration:04d}-prediction.nii.gz"))
        
        

    torchio_logger.info(f"Saved intermediate results to: {sample_dir}")
        


def visualize_3d_slices(image_tensor, label_tensor, pred_tensor, epoch, iteration, vis_dir, torchio_logger):
    """
    Create 3D slice visualizations for input, label, and prediction
    """
    # Convert to numpy and squeeze batch dimension
    if image_tensor.dim() == 5:  # [B, C, D, H, W]
        image = image_tensor[0, 0].cpu().numpy()  # Take first batch, first channel
        label = label_tensor[0, 0].cpu().numpy()  # Take first batch, first channel
        pred = pred_tensor[0, 0].cpu().numpy()    # Take first batch, first channel
    else:
        image = image_tensor[0].cpu().numpy()
        label = label_tensor[0].cpu().numpy()
        pred = pred_tensor[0].cpu().numpy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Epoch {epoch}, Iteration {iteration}', fontsize=16)
    
    # Get middle slices
    d, h, w = image.shape
    mid_d, mid_h, mid_w = d//2, h//2, w//2
    
    # Plot different views
    views = [
        (image[mid_d, :, :], label[mid_d, :, :], pred[mid_d, :, :], 'Sagittal (D)'),
        (image[:, mid_h, :], label[:, mid_h, :], pred[:, mid_h, :], 'Coronal (H)'),
        (image[:, :, mid_w], label[:, :, mid_w], pred[:, :, mid_w], 'Axial (W)')
    ]
    
    for i, (img_slice, lab_slice, pred_slice, title) in enumerate(views):
        # Input image
        axes[0, i].imshow(img_slice, cmap='gray')
        axes[0, i].set_title(f'{title} - Input')
        axes[0, i].axis('off')
        
        # Label and prediction overlay
        axes[1, i].imshow(img_slice, cmap='gray', alpha=0.7)
        axes[1, i].imshow(lab_slice, cmap='Reds', alpha=0.5, label='Ground Truth')
        axes[1, i].imshow(pred_slice, cmap='Blues', alpha=0.5, label='Prediction')
        axes[1, i].set_title(f'{title} - GT (Red) + Pred (Blue)')
        axes[1, i].axis('off')
    
    # Save the plot
    plot_path = os.path.join(vis_dir, 'input_images', f'epoch_{epoch:04d}_iter_{iteration:04d}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    torchio_logger.info(f"Saved visualization to: {plot_path}")
        


def plot_realtime_loss(losses, epoch_losses, plots_dir, train_logger):
    """
    Create and save real-time loss plots
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Iteration-wise losses
    if losses:
        ax1.plot(losses, 'b-', alpha=0.7, linewidth=1)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss (Iteration-wise)')
        ax1.grid(True, alpha=0.3)
        
        # Add moving average
        if len(losses) > 10:
            window_size = min(50, len(losses) // 10)
            moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            ax1.plot(range(window_size-1, len(losses)), moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window_size})')
            ax1.legend()
    
    # Plot 2: Epoch-wise average losses
    if epoch_losses:
        ax2.plot(epoch_losses, 'g-', marker='o', linewidth=2, markersize=6)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Average Loss')
        ax2.set_title('Training Loss (Epoch-wise)')
        ax2.grid(True, alpha=0.3)
        
        # Add trend line
        if len(epoch_losses) > 1:
            z = np.polyfit(range(len(epoch_losses)), epoch_losses, 1)
            p = np.poly1d(z)
            ax2.plot(range(len(epoch_losses)), p(range(len(epoch_losses))), 'r--', alpha=0.8, label='Trend')
            ax2.legend()
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plots_dir, 'realtime_loss.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    # plt.close()
    
    train_logger.info(f"Updated real-time loss plot: {plot_path}")
        


def save_torchio_metrics(predictions, labels, epoch, iteration, intermediate_dir, torchio_logger):
    """
    Calculate and save segmentation metrics using TorchIO
    """
    # Convert to numpy
    pred_np = predictions.cpu().numpy()
    label_np = labels.cpu().numpy()
    
    # Calculate basic metrics
    metrics = {}
    
    # Dice coefficient for each class
    unique_labels = np.unique(label_np)
    dice_scores = []
    
    for label_val in unique_labels:
        if label_val == 0:  # Skip background
            continue
            
        pred_binary = (pred_np == label_val).astype(np.float32)
        label_binary = (label_np == label_val).astype(np.float32)
        
        intersection = np.sum(pred_binary * label_binary)
        union = np.sum(pred_binary) + np.sum(label_binary)
        
        if union > 0:
            dice = 2.0 * intersection / union
            dice_scores.append(dice)
            metrics[f'dice_class_{int(label_val)}'] = dice
    
    # Overall dice score
    if dice_scores:
        metrics['dice_mean'] = np.mean(dice_scores)
    
    # Save metrics to file
    metrics_file = os.path.join(intermediate_dir, 'predictions', f'metrics_epoch_{epoch:04d}_iter_{iteration:04d}.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Epoch: {epoch}, Iteration: {iteration}\n")
        f.write("=" * 40 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    
    torchio_logger.info(f"Saved metrics to: {metrics_file}")
    return metrics
        


def log_model_info(model, model_logger):
    """
    Log detailed model information
    """
    model_logger.info("=" * 50)
    model_logger.info("MODEL ARCHITECTURE INFORMATION")
    model_logger.info("=" * 50)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    model_logger.info(f"Total parameters: {total_params:,}")
    model_logger.info(f"Trainable parameters: {trainable_params:,}")
    model_logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Model structure
    model_logger.info("\nModel structure:")
    model_logger.info(str(model))
    
    # Memory usage
    if torch.cuda.is_available():
        model_logger.info(f"Model device: {next(model.parameters()).device}")
        model_logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        model_logger.info(f"CUDA memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


def log_dataset_info(trainset, testset, torchio_logger):
    """
    Log dataset information using torchio
    """
    torchio_logger.info("=" * 50)
    torchio_logger.info("DATASET INFORMATION")
    torchio_logger.info("=" * 50)
    
    # Training set info
    torchio_logger.info(f"Training set size: {len(trainset.queue_dataset)}")
    if len(trainset.queue_dataset) > 0:
        sample_subject = trainset.queue_dataset[0]
        torchio_logger.info(f"Sample subject keys: {list(sample_subject.keys())}")
        
        for key, value in sample_subject.items():
            if hasattr(value, 'shape'):
                torchio_logger.info(f"  {key} shape: {value.shape}")
            elif hasattr(value, 'data') and hasattr(value.data, 'shape'):
                torchio_logger.info(f"  {key} data shape: {value.data.shape}")
                torchio_logger.info(f"  {key} data type: {value.data.dtype}")
                torchio_logger.info(f"  {key} data range: [{value.data.min():.4f}, {value.data.max():.4f}]")
    
    # Test set info
    torchio_logger.info(f"Test set size: {len(testset.queue_dataset)}")
    
    # Log transformations if available
    if hasattr(trainset, 'transforms') and trainset.transforms:
        torchio_logger.info("Training transforms:")
        for i, transform in enumerate(trainset.transforms):
            torchio_logger.info(f"  {i+1}. {transform}")
    
    if hasattr(testset, 'transforms') and testset.transforms:
        torchio_logger.info("Test transforms:")
        for i, transform in enumerate(testset.transforms):
            torchio_logger.info(f"  {i+1}. {transform}")


def build_accelerator(model, optimizer, trainloader, valloader, warmup_scheduler, training_scheduler):
    accelerator = Accelerator()
    model = accelerator.prepare_model(model=model)
    optimizer = accelerator.prepare_optimizer(optimizer=optimizer)
    trainloader = accelerator.prepare_data_loader(data_loader=trainloader)
    valloader = accelerator.prepare_data_loader(data_loader=valloader)
    warmup_scheduler = accelerator.prepare_scheduler(scheduler=warmup_scheduler)
    training_scheduler = accelerator.prepare_scheduler(scheduler=training_scheduler)

    return accelerator, model, optimizer, trainloader, valloader, warmup_scheduler, training_scheduler

def build_network(args, model_logger):
    model_logger.info("Building network...")
    net = get_model(args)
    
    if args.experiment.ckpt:
        model_logger.info(f"Loading checkpoint from: {args.experiment.ckpt}")
        resume_load_model_checkpoint(net, args)
        model_logger.info("Checkpoint loaded successfully")
    
    return net 

def rebuild_scheduler(args, current_epoch, training_scheduler, warmup_scheduler):
    """_summary_"""
    if args.schedulers.warmup:
        if current_epoch == 0:
            scheduler = warmup_scheduler
        elif current_epoch > args.schedulers.warmup_epoch:
            scheduler = training_scheduler
    elif current_epoch == 0:
        scheduler = training_scheduler
    return scheduler

def train(args):

    ################################################################################
    # Setup logging
    root_logger, train_logger, torchio_logger, model_logger = setup_logging(
        args.experiment.log_path, 
        args.experiment.expname
    )
    
    # Setup visualization directories
    vis_dir, intermediate_dir, plots_dir = setup_visualization_dirs(args.experiment.log_path)
    
    train_logger.info("=" * 80)
    train_logger.info("STARTING TRAINING")
    train_logger.info("=" * 80)
    train_logger.info(f"Experiment: {args.experiment.expname}")
    train_logger.info(f"Log path: {args.experiment.log_path}")
    train_logger.info(f"Visualization dir: {vis_dir}")
    train_logger.info(f"Intermediate results dir: {intermediate_dir}")
    train_logger.info(f"Plots dir: {plots_dir}")
    train_logger.info(f"Start epoch: {args.experiment.start_epoch}")
    train_logger.info(f"Total epochs: {args.experiment.num_epochs}")
    train_logger.info(f"Batch size: {args.dataset.batch}")
    train_logger.info(f"Seed: {args.experiment.seed}")

    model = build_network(args, model_logger)
    log_model_info(model, model_logger)

    # Dataset Creation
    train_logger.info("Building datasets...")
    trainset = build_dataset(args, mode='train')
    testset = build_dataset(args, mode='test')
    
    log_dataset_info(trainset, testset, torchio_logger)
    
    # trainLoader = data.DataLoader(
    #     trainset.queue_dataset, 
    #     batch_size=args.dataset.batch,
    #     shuffle=True, 
    #     pin_memory=False, 
    #     # num_workers=args.dataset.num_workers, 
    #     # persistent_workers=(args.dataset.num_workers>0)
    # )
    # testLoader = data.DataLoader(testset.queue_dataset, 
    #     batch_size=1, 
    #     pin_memory=True, 
    #     shuffle=False, 
    #     num_workers=0)
    
    train_logger.info("Creating TorchIO data loaders...")
    trainLoader = tio.SubjectsLoader(trainset.queue_dataset, batch_size=args.dataset.batch)
    testLoader = tio.SubjectsLoader(testset.queue_dataset, batch_size=1)
    
    torchio_logger.info(f"Train loader created with batch size: {args.dataset.batch}")
    torchio_logger.info(f"Test loader created with batch size: 1")
    torchio_logger.info(f"Number of training batches per epoch: {len(trainLoader)}")
    torchio_logger.info(f"Number of test batches: {len(testLoader)}")

    logging.info(f"Created Dataset and DataLoader")

    ################################################################################
    # Initialize tensorboard, optimizer and etc
    train_logger.info("Initializing TensorBoard writer...")
    writer = SummaryWriter(f"{args.experiment.log_path}")

    train_logger.info("Building optimizer...")
    optimizer = build_optimizer(args.optimizer, model)
    train_logger.info(f"Optimizer: {type(optimizer).__name__}")
    train_logger.info(f"Optimizer parameters: {optimizer.param_groups[0]}")

    train_logger.info("Building schedulers...")
    warmup_scheduler = build_warmup_lr_scheduler(
        args.schedulers, optimizer=optimizer
    )
    training_scheduler = build_training_lr_scheduler(
        args.schedulers, optimizer=optimizer,
    )
    
    train_logger.info("Building loss function...")
    build_loss = SegmentationLossManager()
    loss_fn = build_loss.create_combined_loss(
        args.loss_group.configs, args.loss_group.balanced_weights
    )
    train_logger.info(f"Loss function: {type(loss_fn).__name__}")

    train_logger.info("Building accelerator...")
    accelerator, model, optimizer, trainLoader, testLoader, warmup_scheduler, training_scheduler = build_accelerator(model, optimizer, trainLoader, testLoader, warmup_scheduler, training_scheduler)

    # Now we can use accelerator.is_main_process for all subsequent operations
    if accelerator.is_main_process:
        train_logger.info("Accelerator initialized successfully")
        train_logger.info(f"Number of processes: {accelerator.num_processes}")
        train_logger.info(f"Current process: {accelerator.process_index}")

    if args.ema_group.ema:
        if accelerator.is_main_process:
            train_logger.info("Initializing EMA model...")
        ema_model = torch.optim.swa_utils.AveragedModel(
            model,
            device=accelerator.device,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(
                args.ema_group.ema_decay
            ),
        )
        if accelerator.is_main_process:
            train_logger.info(f"EMA model created with decay: {args.ema_group.ema_decay}")

    ################################################################################
    # Start training
    if accelerator.is_main_process:
        train_logger.info("Starting training loop...")
    elapsed_epochs = 0
    iteration = elapsed_epochs * len(trainLoader)
    
    # Training metrics tracking
    epoch_losses = []
    epoch_times = []
    all_losses = []  # Track all losses for real-time plotting
    all_metrics = []  # Track all metrics
    
    for epoch in range(args.experiment.start_epoch, args.experiment.num_epochs):
        
        if accelerator.is_main_process:
            train_logger.info(f"\n{'='*60}")
            train_logger.info(f"EPOCH {epoch+1}/{args.experiment.num_epochs}")
            train_logger.info(f"{'='*60}")
        
        model.train()

        tic = time.time()
        iter_num_per_epoch = 0 
        epoch_loss = 0.0
        scheduler = rebuild_scheduler(args, epoch, training_scheduler, warmup_scheduler)
        
        if accelerator.is_main_process:
            train_logger.info(f"Using scheduler: {type(scheduler).__name__}")
            train_logger.info(f"Current learning rate: {scheduler.get_last_lr()[0]:.6f}")
        
        for i, inputs in enumerate(trainLoader):

            with accelerator.accumulate(model):
                img, label = inputs['source']['data'], inputs['label']['data']
                img = img.cuda()
                label = label.cuda()

                # Log data information for first batch of first epoch
                if epoch == 0 and i == 0 and accelerator.is_main_process:
                    torchio_logger.info(f"Input image shape: {img.shape}")
                    torchio_logger.info(f"Input label shape: {label.shape}")
                    torchio_logger.info(f"Input image dtype: {img.dtype}")
                    torchio_logger.info(f"Input label dtype: {label.dtype}")
                    torchio_logger.info(f"Input image range: [{img.min():.4f}, {img.max():.4f}]")
                    torchio_logger.info(f"Input label range: [{label.min():.4f}, {label.max():.4f}]")
                    torchio_logger.info(f"Input label unique values: {torch.unique(label)}")

                # uncomment this for visualize the input images and labels for debug

                step = i + epoch * len(trainLoader) # global steps
            
                optimizer.zero_grad()
                

                result = model(img)

                loss = loss_fn(result, label)
                epoch_loss += loss.item()
                all_losses.append(loss.item())  # Track for plotting

                accelerator.backward(loss)
                optimizer.step()


                tic = time.time()
                
                # Only log to TensorBoard on main process
                if accelerator.is_main_process:
                    writer.add_scalar('Train/Loss', loss.item(), iteration+1)

                # Save intermediate results and create visualizations
                if iteration % args.experiment.vis_every == 0:  # Every n iterations
                    # Get predictions (assuming result needs to be processed)

                    if args.ema_group.ema and accelerator.is_main_process:
                        # update_ema_variables(net, ema_net, args.ema_alpha, step)
                        ema_model.update_parameters(model)
                
                    with torch.no_grad():
                        if isinstance(result, (list, tuple)):
                            pred = torch.argmax(result[0], dim=1, keepdim=True)
                        else:
                            pred = torch.argmax(result, dim=1, keepdim=True)
                    
                    # Only save intermediate results on main process
                    if accelerator.is_main_process:
                        # Save TorchIO intermediate results
                        save_torchio_intermediate_results(
                            inputs, pred, epoch, iteration, intermediate_dir, torchio_logger
                        )
                        
                        # Create 3D visualizations
                        visualize_3d_slices(
                            img, label, pred, epoch, iteration, vis_dir, torchio_logger
                        )
                        
                        # Calculate and save metrics
                        metrics = save_torchio_metrics(
                            pred, label, epoch, iteration, intermediate_dir, torchio_logger
                        )
                        all_metrics.append(metrics)
                        
                        # Update real-time loss plot
                        plot_realtime_loss(all_losses, epoch_losses, plots_dir, train_logger)

        

                        # Log every 10 iterations
                        if i % 10 == 0 and accelerator.is_main_process:
                            train_logger.info(
                                f"Epoch {epoch+1}, Iter {i+1}/{len(trainLoader)} -- "
                                f"Loss: {loss.item():.6f} -- "
                                f"LR: {scheduler.get_last_lr()[0]:.6f} -- "
                                f"Step: {iteration}"
                            )
                iteration += 1

        scheduler.step()


        if args.ema_group.ema and (epoch % args.ema_group.val_ema_every == 0) and accelerator.is_main_process:
            train_logger.info("Updating EMA model...")
            temp_ema_model = deepcopy(ema_model).to(
                accelerator.device
            )  # make temp copy
            update_bn(
                trainLoader,
                temp_ema_model,
                device=accelerator.device,
            )
            
        if (epoch % args.experiment.save_every_epochs == 0) and accelerator.is_main_process:
            train_logger.info(f"Saving checkpoint for epoch {epoch}")
            accelerator.save_state(os.path.join(args.experiment.log_path, 'checkpoint', f"checkpoint_{epoch:04d}.pt"), safe_serialization=False)
            if args.ema_group.ema:
                torch.save(temp_ema_model.module, os.path.join(args.experiment.log_path, 'ema', f"checkpoint_{epoch:04d}.pt"))
                train_logger.info(f"EMA checkpoint saved for epoch {epoch}")

        # Log epoch summary
        epoch_time = time.time() - tic
        avg_epoch_loss = epoch_loss / len(trainLoader)
        epoch_losses.append(avg_epoch_loss)
        epoch_times.append(epoch_time)
        
        if accelerator.is_main_process:
            # Calculate epoch metrics
            epoch_metrics = {}
            if all_metrics:
                recent_metrics = all_metrics[-len(trainLoader)//args.experiment.vis_every:]  # Last epoch's metrics
                if recent_metrics:
                    for key in recent_metrics[0].keys():
                        values = [m.get(key, 0) for m in recent_metrics if key in m]
                        if values:
                            epoch_metrics[f'avg_{key}'] = np.mean(values)
            
                train_logger.info(f"\nEpoch {epoch+1} Summary:")
                train_logger.info(f"  Average Loss: {avg_epoch_loss:.6f}")
                train_logger.info(f"  Epoch Time: {epoch_time:.2f} seconds")
                train_logger.info(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
                train_logger.info(f"  Total Iterations: {iteration}")
                
                # Log metrics
                for key, value in epoch_metrics.items():
                    train_logger.info(f"  {key}: {value:.4f}")
                
                # Log to TensorBoard
                writer.add_scalar('Epoch/Average_Loss', avg_epoch_loss, epoch)
                writer.add_scalar('Epoch/Time', epoch_time, epoch)
                writer.add_scalar('Epoch/Learning_Rate', scheduler.get_last_lr()[0], epoch)
                
                # Add metrics to TensorBoard
                for key, value in epoch_metrics.items():
                    writer.add_scalar(f'Epoch/{key}', value, epoch)
                
                # Update final loss plot at end of each epoch
                plot_realtime_loss(all_losses, epoch_losses, plots_dir, train_logger)
                
                # Log memory usage
                if torch.cuda.is_available():
                    train_logger.info(f"  GPU Memory - Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                    train_logger.info(f"  GPU Memory - Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    # Final training summary
    if accelerator.is_main_process:
        train_logger.info(f"\n{'='*80}")
        train_logger.info("TRAINING COMPLETED")
        train_logger.info(f"{'='*80}")
        train_logger.info(f"Total epochs: {args.experiment.num_epochs}")
        train_logger.info(f"Total iterations: {iteration}")
        train_logger.info(f"Final learning rate: {scheduler.get_last_lr()[0]:.6f}")
        train_logger.info(f"Average loss over all epochs: {np.mean(epoch_losses):.6f}")
        train_logger.info(f"Total training time: {sum(epoch_times):.2f} seconds")
        train_logger.info(f"Average time per epoch: {np.mean(epoch_times):.2f} seconds")
        
        # Final metrics summary
        if all_metrics:
            train_logger.info("\nFinal Metrics Summary:")
            final_metrics = {}
            for key in all_metrics[0].keys():
                values = [m.get(key, 0) for m in all_metrics if key in m]
                if values:
                    final_metrics[key] = np.mean(values)
                    train_logger.info(f"  {key}: {np.mean(values):.4f} ± {np.std(values):.4f}")
        
        # Create final comprehensive plots
        train_logger.info("Creating final comprehensive plots...")
        plot_realtime_loss(all_losses, epoch_losses, plots_dir, train_logger)
        
        # Save final loss data
        loss_data = {
            'iteration_losses': all_losses,
            'epoch_losses': epoch_losses,
            'epoch_times': epoch_times,
            'metrics': all_metrics
        }
        np.save(os.path.join(plots_dir, 'training_data.npy'), loss_data)
        train_logger.info(f"Training data saved to: {os.path.join(plots_dir, 'training_data.npy')}")
        
        # Close TensorBoard writer
        writer.close()
        train_logger.info("TensorBoard writer closed")
        
        train_logger.info(f"\nAll results saved to:")
        train_logger.info(f"  Logs: {os.path.join(args.experiment.log_path, 'logs')}")
        train_logger.info(f"  Visualizations: {vis_dir}")
        train_logger.info(f"  Intermediate results: {intermediate_dir}")
        train_logger.info(f"  Plots: {plots_dir}")
        train_logger.info(f"  Checkpoints: {os.path.join(args.experiment.log_path, 'checkpoint')}")


if __name__ == '__main__':


    args = TrainOptions().parse()
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    


    current_time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
    args.experiment.log_path = 'logs/'+args.experiment.expname + '_' + current_time_str


    random.seed(args.experiment.seed)
    np.random.seed(args.experiment.seed)
    torch.manual_seed(args.experiment.seed)

    if hasattr(torch, 'set_deterministic'):
        torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
   
    os.makedirs(args.experiment.log_path, exist_ok=True)
    os.makedirs(os.path.join(args.experiment.log_path, 'checkpoint'), exist_ok=True)
    if args.ema_group.ema:
        os.makedirs(os.path.join(args.experiment.log_path, 'ema'), exist_ok=True)
    save_configure(args)

    train(args)



    
