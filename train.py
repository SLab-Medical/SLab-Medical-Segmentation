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

def rebuild_scheduler(args, epoch, current_epoch, training_scheduler, warmup_scheduler):
    """_summary_"""
    if args.schedulers.warmup:
        if current_epoch == 0:
            scheduler = warmup_scheduler
        elif current_epoch == args.schedulers.warmup_epoch:
            scheduler = training_scheduler
    elif current_epoch == 0:
        scheduler = training_scheduler
    return scheduler

def train(args):

    ################################################################################

    model = build_network(args)


    # Dataset Creation
    trainset = build_dataset(args, mode='train')
    testset = build_dataset(args, mode='test')
    
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
    
    trainLoader = tio.SubjectsLoader(trainset.queue_dataset, batch_size=args.dataset.batch)
    testLoader = tio.SubjectsLoader(testset.queue_dataset, batch_size=1)

    
    logging.info(f"Created Dataset and DataLoader")

    ################################################################################
    # Initialize tensorboard, optimizer and etc
    writer = SummaryWriter(f"{args.experiment.log_path}")

    optimizer = build_optimizer(args.optimizer, model)


    warmup_scheduler = build_warmup_lr_scheduler(
        args.schedulers, optimizer=optimizer
    )
    training_scheduler = build_training_lr_scheduler(
        args.schedulers, optimizer=optimizer,
    )
    
    build_loss = SegmentationLossManager()
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
        scheduler = rebuild_scheduler(args, epoch, epoch, training_scheduler, warmup_scheduler)
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
                loss = loss_fn(result, label)

                accelerator.backward(loss)
                optimizer.step()
                if args.ema_group.ema and (accelerator.is_main_process):
                    # update_ema_variables(net, ema_net, args.ema_alpha, step)
                    ema_model.update_parameters(model)

                tic = time.time()
                

                writer.add_scalar('Train/Loss', loss.item(), iteration+1)

        
        
                iteration += 1


                accelerator.print(
                    f"iter: {str(iteration).zfill(4)} -- "
                    f"train loss: {(loss.item()):.5f} -- "
                    f"lr: {training_scheduler.get_last_lr()[0]}"
                )

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
            if epoch % 1:
                accelerator.save_state(os.path.join(args.experiment.log_path, 'checkpoint', f"checkpoint_{epoch:04d}.pt"), safe_serialization=False)
                if args.ema_group.ema:
                    torch.save(temp_ema_model.module, os.path.join(args.experiment.log_path, 'ema', f"checkpoint_{epoch:04d}.pt"))

            



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



    
