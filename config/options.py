import argparse
import os
import torch
import yaml
from munch import *

def merge_munch(m2: Munch, opt: Munch) -> Munch:
    for k, v in m2.items():
        if k not in opt:
            opt[k] = v
        else:
            if isinstance(v, Munch) and isinstance(opt[k], Munch):
                merge_munch(v, opt[k])
    return opt

class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Medical Image Segmentation')
        self.initialized = False

    def initialize(self):
        # Dataset options
        dataset = self.parser.add_argument_group('dataset')
        dataset.add_argument("--dataset_name", type=str, default='liuxiangyue', help="dataset name")
        dataset.add_argument("--dataset_path", type=str, default='/opt/data/private/songshuang/med/stylesdf/All_Image_Esophagus_Data', help="path to the dataset")
        dataset.add_argument("--batch", type=int, default=4, help="batchsize")
        dataset.add_argument("--num_workers", type=int, default=4, help="num_workers")
        dataset.add_argument("--drop_last", type=bool, default=True, help="drop_last")
        dataset.add_argument("--queue_length", type=int, default=5, help="queue_length")
        dataset.add_argument("--samples_per_volume", type=int, default=5, help="samples_per_volume")
        dataset.add_argument("--patch_size", nargs='+', type=int, default=[64, 64, 64], help="patch_size")
                           
        
        # Experiment Options
        experiment = self.parser.add_argument_group('experiment')
        experiment.add_argument("--expname", type=str, default='debug', help='experiment name')
        experiment.add_argument("--ckpt", type=str, default=None, help="path to the checkpoints to resume training")
        experiment.add_argument("--conf", type=str, default=None, help="path to the config yaml")
        experiment.add_argument("--continue_training", action="store_true", help="continue training the model")
        experiment.add_argument("--seed", type=int, default=666, help="seed")
        experiment.add_argument("--start_epoch", type=int, default=0, help="start_epoch")
        experiment.add_argument("--num_epochs", type=int, default=20, help="num_epochs")
        experiment.add_argument("--save_every_epochs", type=int, default=2, help="save_every_epochs")
        experiment.add_argument("--vis_every", type=int, default=200, help="print_every")
        experiment.add_argument("--grad_accumulate_steps", type=int, default=1, help="grad_accumulate_steps")

        # Optimizer Options
        optimizer = self.parser.add_argument_group('optimizer')
        optimizer.add_argument("--optimizer_type", type=str, default='adamw', help='optimizer_type')
        optimizer.add_argument("--lr", type=float, default=0.0001, help='learning rate')


        # Schedulers Options
        schedulers = self.parser.add_argument_group('schedulers')
        schedulers.add_argument("--scheduler_type", type=str, default='cosine_annealing_wr', help='scheduler_type')
        schedulers.add_argument("--warmup", type=bool, default=True, help='warmup enabled?')
        schedulers.add_argument("--warmup_epoch", type=int, default=20, help='warmup_epoch')



        # Model options
        model = self.parser.add_argument_group('model')
        model.add_argument("--model_name", type=str, default='EfficientMedNeXt_L', help='checkpoints directory name')
        model.add_argument("--dimension", type=str, default='3d', help='model dimension')
        model.add_argument("--in_channels", type=int, default=1, help='in_channels')
        model.add_argument("--num_classes", type=int, default=1, help='num_classes')


        # Loss options
        loss = self.parser.add_argument_group('loss_group')
        loss.add_argument("--loss_type", 
                          type=str, 
                          default='combine', 
                          help='loss choices, selected from choices')
        

        # Ema options
        ema = self.parser.add_argument_group('ema_group')
        ema.add_argument("--ema", type=bool, default=True, help="ema enabled?")
        ema.add_argument("--ema_decay", type=float, default=0.999, help="ema_decay")
        ema.add_argument("--val_ema_every", type=int, default=1, help="val_ema_every")

        self.initialized = True

    def parse(self):
        self.opt = Munch()
        if not self.initialized:
            self.initialize()

        args = self.parser.parse_args()

        for group in self.parser._action_groups[2:]:
            title = group.title
            self.opt[title] = Munch()
            for action in group._group_actions:
                dest = action.dest
                self.opt[title][dest] = args.__getattribute__(dest)


            if title == 'optimizer':
                optim_config_path = 'config/optimizer/%s.yaml'%(self.opt[title]['optimizer_type'])
                with open(optim_config_path, 'r') as f:
                    optim_config = yaml.load(f, Loader=yaml.SafeLoader)
                # for key, value in optim_config.items():
                #     if not hasattr(self.opt[title], key) or getattr(self.opt[title], key) in [None, False, '', 0]:
                #         setattr(self.opt[title], key, value)
                m_optim = munchify(optim_config)
                self.opt = merge_munch(m_optim, self.opt)
            elif title == 'schedulers':
                sche_config_path = 'config/scheduler/%s.yaml'%(self.opt[title]['scheduler_type'])
                with open(sche_config_path, 'r') as f:
                    sche_config = yaml.load(f, Loader=yaml.SafeLoader)
                # for key, value in sche_config.items():
                #     if not hasattr(self.opt[title], key) or getattr(self.opt[title], key) in [None, False, '', 0]:
                #         setattr(self.opt[title], key, value)
                m_she = munchify(sche_config)
                self.opt = merge_munch(m_she, self.opt)
            elif title == 'model':
                arch_config_path = 'config/arch/%s.yaml'%(self.opt[title]['model_name'])
                with open(arch_config_path, 'r') as f:
                    arch_config = yaml.load(f, Loader=yaml.SafeLoader)
                # for key, value in arch_config.items():
                #     if not hasattr(self.opt[title], key) or getattr(self.opt[title], key) in [None, False, '', 0]:
                #         setattr(self.opt[title], key, value)
                m_arch = munchify(arch_config)
                self.opt = merge_munch(m_arch, self.opt)
            elif title == 'loss_group':
                # loss_type_str = '-'.join(self.opt[title]['loss_type'])
                # loss_config_path = 'config/losses/%s.yaml'%(loss_type_str) 
                # with open(loss_config_path, 'r') as f:
                #     loss_config = yaml.load(f, Loader=yaml.SafeLoader)
                # for key, value in loss_config.items():
                #     if not hasattr(self.opt[title], key) or getattr(self.opt[title], key) in [None, False, '', 0]:
                #         setattr(self.opt[title], key, value)  
                with open('config/losses/%s.yaml'%(self.opt[title]['loss_type'])) as f:
                    loss_config = yaml.safe_load(f)
                m_loss = munchify(loss_config)
                self.opt = merge_munch(m_loss, self.opt)

        if self.opt.experiment.conf is not None:
            with open(self.opt.experiment.conf) as f:
                exp_config = yaml.safe_load(f)
            m_exp = munchify(exp_config)
            self.opt = merge_munch(m_exp, self.opt)
        return self.opt