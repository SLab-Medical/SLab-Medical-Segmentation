import os
import logging
import torch
import torch.distributed as dist
import pdb

def resume_load_model_checkpoint(net, ema_net, args):
    assert args.load != False, "Please specify the load path with --load"
    
    checkpoint = torch.load(args.load)
    net.load_state_dict(checkpoint['model_state_dict'])
    args.start_epoch = checkpoint['epoch']

    if args.ema:
        ema_net.load_state_dict(checkpoint['ema_model_state_dict'])

def resume_load_optimizer_checkpoint(optimizer, args):
    assert args.load != False, "Please specify the load path with --load"
    
    checkpoint = torch.load(args.load)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])