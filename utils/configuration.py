import os
import logging
import torch
import torch.distributed as dist
import pdb
import yaml
def is_master(args):
    return args.rank % args.ngpus_per_node == 0

def save_configure(args):
    with open(os.path.join(args.experiment.log_path,'config.yaml'), 'w') as f:
        yaml.safe_dump(args, f)