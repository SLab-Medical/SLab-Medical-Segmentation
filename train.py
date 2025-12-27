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

from utils.configuration import save_configure
from utils.resume import resume_load_model_checkpoint, resume_load_optimizer_checkpoint
from utils.ema import update_ema_variables, update_bn
from utils.plot_metrics import MetricsPlotter
from utils.metrics import calculate_metrics

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


def build_accelerator(model, optimizer, trainloader, valloader, warmup_scheduler, training_scheduler):
    accelerator = Accelerator()
    model = accelerator.prepare_model(model=model)
    optimizer = accelerator.prepare_optimizer(optimizer=optimizer)
    trainloader = accelerator.prepare_data_loader(data_loader=trainloader)
    valloader = accelerator.prepare_data_loader(data_loader=valloader)
    warmup_scheduler = accelerator.prepare_scheduler(scheduler=warmup_scheduler)
    training_scheduler = accelerator.prepare_scheduler(scheduler=training_scheduler)

    return accelerator, model, optimizer, trainloader, valloader, warmup_scheduler, training_scheduler

def build_network(args):
    net = get_model(args)
    
    if args.experiment.ckpt:
        resume_load_model_checkpoint(net, args)
    
    

    return net 

def rebuild_scheduler(args, current_epoch, training_scheduler, warmup_scheduler):
    """_summary_"""
    if args.schedulers.warmup:
        if current_epoch < args.schedulers.warmup_epoch:
            scheduler = warmup_scheduler
        elif current_epoch >= args.schedulers.warmup_epoch:
            scheduler = training_scheduler
    else:
        scheduler = training_scheduler
    return scheduler

def save_intermediate_tio(batch_inputs, logits, base_dir, epoch, iteration, is_2d=False):
    os.makedirs(base_dir, exist_ok=True)
    # save only the first item in the batch to limit IO
    img_tensor = batch_inputs['source']['data'][0].detach().cpu().float()
    lbl_tensor = batch_inputs['label']['data'][0].detach().cpu()

    pred_logits = logits.detach().cpu()[0]
    if pred_logits.shape[0] > 1:
        # multi-class: argmax over channel
        pred_tensor = torch.argmax(pred_logits, dim=0, keepdim=True).to(torch.int16)
    else:
        # binary: sigmoid then threshold
        pred_tensor = (torch.sigmoid(pred_logits) > 0.5).to(torch.int16)

    if is_2d:
        # Save as PNG for 2D data
        from PIL import Image
        img_path = os.path.join(base_dir, f"epoch{epoch:04d}_iter{iteration:06d}_img.png")
        gt_path = os.path.join(base_dir, f"epoch{epoch:04d}_iter{iteration:06d}_gt.png")
        pr_path = os.path.join(base_dir, f"epoch{epoch:04d}_iter{iteration:06d}_pred.png")

        # Convert tensors to numpy and normalize for visualization
        img_np = (img_tensor[0].numpy() * 255).astype(np.uint8)
        gt_np = (lbl_tensor[0].numpy() * 255).astype(np.uint8)
        pred_np = (pred_tensor[0].numpy() * 255).astype(np.uint8)

        Image.fromarray(img_np).save(img_path)
        Image.fromarray(gt_np).save(gt_path)
        Image.fromarray(pred_np).save(pr_path)
    else:
        # Save as NIfTI for 3D data
        affine = np.eye(4)
        img_path = os.path.join(base_dir, f"epoch{epoch:04d}_iter{iteration:06d}_img.nii.gz")
        gt_path = os.path.join(base_dir, f"epoch{epoch:04d}_iter{iteration:06d}_gt.nii.gz")
        pr_path = os.path.join(base_dir, f"epoch{epoch:04d}_iter{iteration:06d}_pred.nii.gz")

        tio.ScalarImage(tensor=img_tensor, affine=affine).save(img_path)
        tio.LabelMap(tensor=lbl_tensor.to(torch.int16), affine=affine).save(gt_path)
        tio.LabelMap(tensor=pred_tensor, affine=affine).save(pr_path)

def train(args):

    ################################################################################

    model = build_network(args)


    # Dataset Creation
    trainset = build_dataset(args, mode='train')
    testset = build_dataset(args, mode='test')

    # Check if we're using 2D or 3D data
    is_2d = args.model.dimension == '2d'

    if is_2d:
        # Use standard DataLoader for 2D data
        trainLoader = data.DataLoader(
            trainset.queue_dataset,
            batch_size=args.dataset.batch,
            shuffle=True,
            pin_memory=True,
            num_workers=args.dataset.num_workers if hasattr(args.dataset, 'num_workers') else 4,
        )
        testLoader = data.DataLoader(
            testset.queue_dataset,
            batch_size=1,
            pin_memory=True,
            shuffle=False,
            num_workers=0
        )
    else:
        # Use TorchIO SubjectsLoader for 3D data
        trainLoader = tio.SubjectsLoader(trainset.queue_dataset, batch_size=args.dataset.batch)
        testLoader = tio.SubjectsLoader(testset.queue_dataset, batch_size=1)

    
    logging.info(f"Created Dataset and DataLoader")

    ################################################################################
    # Initialize tensorboard, optimizer and etc
    writer = SummaryWriter(f"{args.experiment.log_path}")

    # Initialize metrics plotter for real-time visualization
    plot_dir = os.path.join(args.experiment.log_path, 'plots')
    plotter = MetricsPlotter(save_dir=plot_dir, window_size=50)

    optimizer = build_optimizer(args.optimizer, model)


    warmup_scheduler = build_warmup_lr_scheduler(
        args.schedulers, optimizer=optimizer
    )
    training_scheduler = build_training_lr_scheduler(
        args.schedulers, optimizer=optimizer,
    )

    # Create loss function with automatic parameter adjustment based on num_classes
    num_classes = args.model.out_channels
    build_loss = SegmentationLossManager(num_classes=num_classes)
    loss_fn = build_loss.create_combined_loss(
        args.loss_group.configs, args.loss_group.balanced_weights
    )

    accelerator, model, optimizer, trainLoader, testLoader, warmup_scheduler, training_scheduler = build_accelerator(model, optimizer, trainLoader, testLoader, warmup_scheduler, training_scheduler)

    if args.ema_group.ema:
        ema_model = torch.optim.swa_utils.AveragedModel(
            model,
            device=accelerator.device,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(
                args.ema_group.ema_decay
            ),
        )

    ################################################################################
    # Start training
    elapsed_epochs = 0
    iteration = elapsed_epochs * len(trainLoader)
    
    for epoch in range(args.experiment.start_epoch, args.experiment.num_epochs):
        
        model.train()

        tic = time.time()
        iter_num_per_epoch = 0 
        scheduler = rebuild_scheduler(args, epoch, training_scheduler, warmup_scheduler)
        for i, inputs in enumerate(trainLoader):

            with accelerator.accumulate(model):
                img, label = inputs['source']['data'], inputs['label']['data']
                img = img.cuda()
                label = label.cuda()

            
                # uncomment this for visualize the input images and labels for debug

                step = i + epoch * len(trainLoader) # global steps
            
                optimizer.zero_grad()
                

                result = model(img)
                if isinstance(result, (list, tuple)):
                    result = result[0]
                if result.shape[2:] != label.shape[2:]:
                    # Use appropriate interpolation mode based on dimension
                    if is_2d:
                        result = F.interpolate(
                            result,
                            size=label.shape[2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    else:
                        result = F.interpolate(
                            result,
                            size=label.shape[2:],
                            mode="trilinear",
                            align_corners=False,
                        )
                loss = loss_fn(result, label)

                # Calculate comprehensive metrics - supports both binary and multi-class
                # Distance metrics (Hausdorff, ASD) are optional as they're computationally expensive
                include_distance = getattr(args.experiment, 'include_distance_metrics', False)
                metrics = calculate_metrics(result, label, num_classes=num_classes,
                                          threshold=0.5, include_distance_metrics=include_distance)

                accelerator.backward(loss)
                optimizer.step()
                if args.ema_group.ema and (accelerator.is_main_process):
                    # update_ema_variables(net, ema_net, args.ema_alpha, step)
                    ema_model.update_parameters(model)

                tic = time.time()

                if accelerator.is_main_process:
                    # Log to tensorboard - Core metrics
                    writer.add_scalar('Train/Loss/total', loss.item(), iteration + 1)
                    writer.add_scalar('Train/Metrics/dice', metrics['dice'], iteration + 1)
                    writer.add_scalar('Train/Metrics/iou', metrics['iou'], iteration + 1)
                    writer.add_scalar('Train/Metrics/precision', metrics['precision'], iteration + 1)
                    writer.add_scalar('Train/Metrics/recall', metrics['recall'], iteration + 1)
                    writer.add_scalar('Train/Metrics/specificity', metrics['specificity'], iteration + 1)
                    writer.add_scalar('Train/Metrics/f1', metrics['f1'], iteration + 1)
                    writer.add_scalar('Train/Metrics/accuracy', metrics['accuracy'], iteration + 1)

                    # Log distance metrics if enabled
                    if include_distance and 'hausdorff_distance' in metrics:
                        writer.add_scalar('Train/Metrics/hausdorff_distance', metrics['hausdorff_distance'], iteration + 1)
                    if include_distance and 'avg_surface_distance' in metrics:
                        writer.add_scalar('Train/Metrics/avg_surface_distance', metrics['avg_surface_distance'], iteration + 1)

                    if hasattr(loss_fn, 'components'):
                        comps = loss_fn.components()
                        for k, v in comps.items():
                            writer.add_scalar(f'Train/Loss/{k}', float(v.item()), iteration + 1)

                    # Update metrics plotter with all metrics
                    metric_dict = {
                        'total_loss': loss.item(),
                        'dice': metrics['dice'],
                        'iou': metrics['iou'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'specificity': metrics['specificity'],
                        'f1': metrics['f1'],
                        'accuracy': metrics['accuracy']
                    }
                    # Add distance metrics if enabled
                    if include_distance and 'hausdorff_distance' in metrics:
                        metric_dict['hausdorff_distance'] = metrics['hausdorff_distance']
                    if include_distance and 'avg_surface_distance' in metrics:
                        metric_dict['avg_surface_distance'] = metrics['avg_surface_distance']

                    if hasattr(loss_fn, 'components'):
                        comps = loss_fn.components()
                        for k, v in comps.items():
                            metric_dict[k + '_loss'] = float(v.item())

                    plotter.update(iteration + 1, **metric_dict)

                    # Generate and save plots every 20 iterations
                    if (iteration + 1) % 20 == 0:
                        plotter.plot(show_ma=True)
                        plotter.save_csv()

        
        
                iteration += 1


                # Print loss information
                loss_msg = f"Epoch: {epoch+1:03d}/{args.experiment.num_epochs:03d} | \
                            iter: {str(iteration).zfill(6)} | \
                            Loss - total: {loss.item():.5f}"
                if hasattr(loss_fn, 'components'):
                    comps = loss_fn.components()
                    comp_str = ' | '.join([f"{k}: {float(v.item()):.5f}" for k, v in comps.items()])
                    loss_msg = loss_msg + ' | ' + comp_str
                accelerator.print(loss_msg)

                # Print metrics information (separate line)
                metrics_msg = f"{'':>46} Metrics - Dice: {metrics['dice']:.4f} | \
                            IoU: {metrics['iou']:.4f} | \
                            F1: {metrics['f1']:.4f} | \
                            Acc: {metrics['accuracy']:.4f} | \
                            Precision: {metrics['precision']:.4f} | \
                            Recall: {metrics['recall']:.4f}"
                accelerator.print(metrics_msg)

                if accelerator.is_main_process and (iteration % max(1, args.experiment.vis_every) == 0):
                    inter_dir = os.path.join(args.experiment.log_path, 'intermediate')
                    save_intermediate_tio(inputs, result, inter_dir, epoch, iteration, is_2d=is_2d)


            scheduler.step()


            if args.ema_group.ema and (epoch % args.ema_group.val_ema_every == 0):
                temp_ema_model = deepcopy(ema_model).to(
                    accelerator.device
                )  # make temp copy
                update_bn(
                    trainLoader,
                    temp_ema_model,
                    device=accelerator.device,
                )
            if epoch % args.experiment.save_every_epochs == 0:
                accelerator.save_state(os.path.join(args.experiment.log_path, 'checkpoint', f"checkpoint_{epoch:04d}.pt"), safe_serialization=False)
                if args.ema_group.ema:
                    torch.save(temp_ema_model.module.state_dict(), os.path.join(args.experiment.log_path, 'ema', f"checkpoint_{epoch:04d}.pt"))

            



if __name__ == '__main__':

    # accelerator = Accelerator()

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



    
