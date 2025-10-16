import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import optim

def build_optimizer(args, net):
    if args.optimizer_type == 'sgd':
        # TODO
        return optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer_type == 'adam':
        # TODO
        return optim.Adam(net.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)
    elif args.optimizer_type == 'adamw':
        return optim.AdamW(net.parameters(), lr=args.lr, betas=args.optim_betas, weight_decay=args.optim_weight_decay, eps=1e-5) # larger eps has better stability during AMP training

        