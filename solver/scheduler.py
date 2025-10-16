import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import optim

def build_warmup_lr_scheduler(config, optimizer):
    """
    Linearly ramps up the learning rate within X
    number of epochs to the working epoch.
    Args:
        optimizer (_type_): _description_
        warmup_epochs (_type_): _description_
        warmup_lr (_type_): warmup lr should be the starting lr we want.
    """
    lambda1 = lambda epoch: (
        (epoch + 1) * 1.0 / config["warmup_epoch"]
    )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1, verbose=False)
    return scheduler

def build_training_lr_scheduler(config, optimizer):
    """
    Wraps a normal scheuler
    """
    scheduler_type = config["scheduler_type"]
    if scheduler_type == "reducelronplateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.1,
            mode=config["train_scheduler"]["mode"],
            patience=config["train_scheduler"]["patience"],
            verbose=False,
            min_lr=config["train_scheduler"]["scheduler_args"]["min_lr"],
        )
        return scheduler
    elif scheduler_type == "cosine_annealing_wr":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config["sche_t_0_epochs"],
            T_mult=config["sche_t_mult"],
            eta_min=config["sche_min_lr"],
            last_epoch=-1,
            verbose=False,
        )
        return scheduler
    elif scheduler_type == "poly_lr":
        scheduler = optim.lr_scheduler.PolynomialLR(
            optimizer=optimizer,
            total_iters=5,
            power=config["train_scheduler"]["scheduler_args"]["power"],
            last_epoch=-1,
        )
        return scheduler
    else:
        raise NotImplementedError("Specified Scheduler Is Not Implemented")