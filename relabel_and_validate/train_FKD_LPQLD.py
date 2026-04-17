"""
code adapted from: https://github.com/VILA-Lab/SRe2L/blob/main/SRe2L/validate/train_FKD.py
"""
import argparse
import math
import os
import sys
import random
import shutil
import time
from collections import defaultdict
import json
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms import InterpolationMode
from utils import AverageMeter, accuracy, get_parameters, str2bool, print_args_rich
import utils_pruning

from typing import NamedTuple

import wandb
from wandb_utils.wandb_logger import wandb_log, wandb_hyperparam_log, wandb_terminal_log

from torch.utils.data import Sampler

# import px
import plotly.express as px
import matplotlib.pyplot as plt

sys.path.append('..')
# It is imported for you to access and modify the PyTorch source code (via Ctrl+Click), more details in README.md
from torch.utils.data._utils.fetch import _MapDatasetFetcher

from utils_fkd import (ComposeWithCoords, ImageFolder_FKD_MIX_BATCH, MultiDatasetImageFolder,
                               RandomHorizontalFlipWithRes,
                               RandomResizedCropWithCoords, mix_aug, 
                               ComposeWithCoords_Cutout,
                               ComposeWithCoords_RandAug,
                               CutoutPILWithCoords,
                               get_class_distribution)
from rand_aug import rand_augment_transform_with_exact_params

from tqdm import trange

from ema_pytorch import EMA

import warnings

# ignore warnings
warnings.filterwarnings("ignore", category=UserWarning)


def get_args():
    parser = argparse.ArgumentParser("FKD Training on ImageNet-1K")
    parser.add_argument('--dataset', type=str, default='imagenet1k', choices=['imagenet1k', 'imagenet21k', 'tiny'],)
    parser.add_argument('--batch_size', type=int,
                        default=1024, help='batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int,
                        default=1, help='gradient accumulation steps for small gpu memory')
    parser.add_argument('--start-epoch', type=int,
                        default=0, help='start epoch')
    parser.add_argument('--epochs', type=int, default=300, help='total epoch')
    parser.add_argument('-j', '--workers', default=16, type=int,
                        help='number of data loading workers')

    parser.add_argument('--train_dir', type=str, default=None,
                        help='path to training dataset')
    parser.add_argument('--val_dir', type=str,
                        default='/path/to/imagenet/val', help='path to validation dataset')
    parser.add_argument('--output_dir', type=str,
                        default='./save/1024', help='path to output dir')

    parser.add_argument('--cos', default=False,
                        action='store_true', help='cosine lr scheduler')
    parser.add_argument('--sgd', default=False,
                        action='store_true', help='sgd optimizer')
    parser.add_argument('-lr', '--learning-rate', type=float,
                        default=1.024, help='sgd init learning rate')  # checked
    parser.add_argument('--momentum', type=float,
                        default=0.875, help='sgd momentum')  # checked
    parser.add_argument('--weight-decay', type=float,
                        default=3e-5, help='sgd weight decay')  # checked
    parser.add_argument('--adamw-lr', type=float,
                        default=0.001, help='adamw learning rate')
    parser.add_argument('--adamw-weight-decay', type=float,
                        default=0.01, help='adamw weight decay')

    parser.add_argument('--model', type=str,
                        default='resnet18', help='student model name')

    parser.add_argument('--keep-topk', type=int, default=1000,
                        help='keep topk logits for kd loss')
    parser.add_argument('-T', '--temperature', type=float,
                        default=20.0, help='temperature for distillation loss')
    parser.add_argument('--fkd_path', type=str,
                        default=None, help='path to fkd label')
    parser.add_argument('--mix_type', default=None, type=str,
                        choices=['mixup', 'cutmix', None], help='mixup or cutmix or None')
    parser.add_argument('--fkd_seed', default=42, type=int,
                        help='seed for batch loading sampler')
    parser.add_argument('--val_interval', default=50, type=int, help='validation interval')
    
    # data pruning configs
    parser.add_argument('--prune_label', type=bool, default=True, help='prune label')
    parser.add_argument('--prune_ratio', type=float, default=0.5, help='prune away `prune_ratio` of data')
    parser.add_argument('--sample_metric', type=str, default='random', choices=['random', 'order'], help='sampling method')

    # removed granularity parameter since we only use batch_to_epoch approach
    parser.add_argument('--smoothing_lr', type=str2bool, default=False, help='use smoothing lr scheduler')
    parser.add_argument('--smoothing_lr_strength', type=float, default=2, help='strength of smoothing lr scheduler')
    parser.add_argument('--ema', type=str2bool, default=False, help='whether to use ema model to validate')
    parser.add_argument('--ema_decay', type=float, default=0.99, help='ema decay rate')

    parser.add_argument("--min-scale-crops", type=float, default=0.08,
                        help="argument in RandomResizedCrop")
    # imagent21k-P configs
    parser.add_argument('--label_smoothing', type=float, default=0, help='label smoothing')
    
    # Label Quantization (Q): stores only top-k pre-softmax logits to reduce storage
    # while enabling augmentation-per-image diversity (paper Sec. III-C).
    parser.add_argument('--label_quantization', type=str, default=None,
                        help='Label Quantization method (paper Sec. III-C). '
                             '"MR-k": Marginal Re-normalization — keep top-k logits, normalize to sum=1 (recommended); '
                             '"MS-k": Marginal Smoothing — keep top-k, distribute remaining mass uniformly; '
                             '"hard": convert to one-hot.')
    
    # wandb logger
    parser.add_argument('--use_wandb', default=True, action='store_false', help='use wandb logger')
    parser.add_argument('--run_name', type=str, default="default", help='name of the run')
    parser.add_argument('--exp_name', type=str, default="label_pruning", help='name of the experiment')
    parser.add_argument('--tag_name', type=str, default=None, help='tag of the run')
    parser.add_argument('--cfg_yaml', type=str, default=None, help='path to config file')
    
    parser.add_argument('--gpus', type=str, default='0', help='visible devices')

    parser.add_argument('--train_epochs', type=int, default=-1, help='specifying the number of training epochs')

    # New arguments for batch selection strategy
    parser.add_argument('--batch_selection_strategy', type=str, default='random',
                        choices=['random', 'metric_ranking'],
                        help='Strategy for selecting batches: random or metric_ranking')
    parser.add_argument('--batch_selection_metric', type=str, default=None,
                        help='Metric to use for ranking batches (e.g., max_per_sample_mean, top1_class_count)')
    parser.add_argument('--batch_selection_order', type=str, default='descending',
                        choices=['ascending', 'descending', 'middle'],
                        help='How to sort batches: ascending, descending, or middle')

    # Dynamic Knowledge Reuse (DKR): temperature annealing on stored logits to extract
    # diverse supervisory signals across training epochs (Sec. III-D of LPQLD paper).
    parser.add_argument('--temp_scheduler', type=str, default=None,
                        choices=['step', 'cosine', 'step_reverse', 'cosine_reverse', None],
                        help='DKR temperature scheduler type (paper Sec. III-D). '
                             '"step": decay by temp_step_gamma every temp_step_size epochs; '
                             '"cosine": cosine annealing to temp_min.')
    parser.add_argument('--temp_step_size', type=int, default=30,
                        help='DKR step size: decay temperature every N epochs (used with --temp_scheduler step)')
    parser.add_argument('--temp_step_gamma', type=float, default=0.5,
                        help='DKR decay factor per step (used with --temp_scheduler step)')
    parser.add_argument('--temp_min', type=float, default=1.0,
                        help='DKR minimum temperature for cosine scheduler (used with --temp_scheduler cosine)')
    parser.add_argument('--temp_stu', type=float, default=-1.0,
                        help='Fixed student temperature for Calibrated Student-Teacher Alignment (CA, paper Sec. III-E). '
                             'Disabled if <= 0.')
    parser.add_argument('--temp_stu_dynamic', type=float, default=-1.0,
                        help='Dynamic student temperature scale for Calibrated Alignment (CA, paper Sec. III-E). '
                             'Grid-searched ratio applied to teacher temperature. Disabled if <= 0.')
    
    # store true
    parser.add_argument('--use_rand_aug', default=False, action='store_true', help='use rand aug')
    parser.add_argument('--rand_augment_config', type=str, default='rand-m6-n2-mstd1.0', help='rand aug policy')
    
    args = parser.parse_args()
    
    if args.cfg_yaml:
        # load yaml config
        import yaml

        def load_config(config_file):
            with open(config_file, 'r') as file:
                config = yaml.safe_load(file)
            return config

        # Load the YAML configuration, for example:
        cfg = load_config(args.cfg_yaml)

        # set key-value
        cfg_keys = ['basic', 'path']
        for cfg_key in cfg_keys:
            for key in cfg['validate'][cfg_key].keys():
                setattr(args, key, cfg['validate'][cfg_key][key])
        
        # set store_true args
        for key in cfg['validate']['store_true']:
            setattr(args, key, True)
        
        # shared config
        # check if common config exists
        if cfg.get('common') is not None:
            common_keys = ['prune', 'basic', 'path']
            for common_key in common_keys:
                if cfg['common'].get(common_key) is None:
                    continue
                for key in cfg['common'][common_key].keys():
                    setattr(args, key, cfg['common'][common_key][key])

        args.output_dir = args.output_dir + f'_T[{int(args.temperature)}]'
        args.run_name = args.run_name + f'_T[{int(args.temperature)}]'

        # Add temperature scheduler info to output_dir and run_name if it exists
        if args.temp_scheduler:
            if args.temp_scheduler == 'step':
                temp_sched_info = f'_TS[step_{args.temp_step_size}_{args.temp_step_gamma}]'
            elif args.temp_scheduler == 'cosine':
                temp_sched_info = f'_TS[cos_{args.temp_min}]'
            elif args.temp_scheduler == 'step_reverse':
                temp_sched_info = f'_TS[step_rev_{args.temp_step_size}_{args.temp_step_gamma}]'
            elif args.temp_scheduler == 'cosine_reverse':
                temp_sched_info = f'_TS[cos_rev_{args.temp_min}]'
            
            if args.temp_stu_dynamic > 0:
                temp_sched_info += f'_stu_dynamic[{args.temp_stu_dynamic}]'
            elif args.temp_stu > 0:
                temp_sched_info += f'_stu[{args.temp_stu}]'

            args.output_dir = args.output_dir + temp_sched_info
            args.run_name = args.run_name + temp_sched_info

        # add prune config taggings
        if args.prune_label:
            if args.batch_selection_strategy == 'random':
                suffix = f'_M[random]'

            suffix += f'_R[{args.prune_ratio}]'
        
            if args.train_epochs > 0:
                suffix += f'_EP[{args.train_epochs}]'
            
            if args.adamw_lr != 0.001:
                suffix += f'_LR[{args.adamw_lr}]'

            args.output_dir = args.output_dir + suffix
            args.run_name = args.run_name + suffix
    

    args.cur_time = time.strftime("%Y%m%d-%H%M%S")

    args.mode = 'fkd_load'
    return args

def load_dataset(args):
    # Data loading
    if args.dataset == 'imagenet1k':
        image_size = 224
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        if args.use_rand_aug:
            train_transform = ComposeWithCoords_RandAug(transforms=[
                        rand_augment_transform_with_exact_params(
                            config_str=args.rand_augment_config, 
                            hparams={'translate_const': 117, 'img_mean': (124, 116, 104)}
                        ),
                        RandomResizedCropWithCoords(size=image_size,
                                                    scale=(args.min_scale_crops, 1),
                                                    interpolation=InterpolationMode.BILINEAR),
                        RandomHorizontalFlipWithRes(),
                        transforms.ToTensor(),
                        normalize,
                    ])
        else:
            train_transform = ComposeWithCoords(transforms=[
                        RandomResizedCropWithCoords(size=image_size,
                                                    scale=(args.min_scale_crops, 1),
                                                    interpolation=InterpolationMode.BILINEAR),
                        RandomHorizontalFlipWithRes(),
                        transforms.ToTensor(),
                        normalize,
                    ])

        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    elif args.dataset == 'tiny':
        assert args.use_rand_aug is False, "rand aug is not supported for tiny"

        # https://github.com/zeyuanyin/tiny-imagenet/blob/main/classification/train.py
        image_size = 64
        normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                    std=[0.2302, 0.2265, 0.2262])
        train_transform = ComposeWithCoords(transforms=[
                    RandomResizedCropWithCoords(size=image_size,
                                                scale=(args.min_scale_crops, 1),
                                                interpolation=InterpolationMode.BILINEAR),
                    RandomHorizontalFlipWithRes(),
                    transforms.ToTensor(),
                    normalize,
                ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    elif args.dataset == 'imagenet21k':
        assert args.use_rand_aug is False, "rand aug is not supported for imagenet21k"
        
        image_size = 224
        normalize = None    # no normalization is used for trianing imagenet21k-P

        train_transform = ComposeWithCoords_Cutout(transforms=[
                RandomResizedCropWithCoords(size=image_size,
                                            scale=(0.08, 1),
                                            interpolation=InterpolationMode.BILINEAR),
                CutoutPILWithCoords(cutout_factor=0.5),
                transforms.ToTensor(),
            ])
        
        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    batch_indices_path = os.path.join(args.fkd_path, 'sampler_indices_batch.json')
    train_dataset = ImageFolder_FKD_MIX_BATCH(
        fkd_path=args.fkd_path,
        mode=args.mode,
        args_epoch=args.epochs,
        args_bs=args.batch_size,
        root=args.train_dir,
        batch_to_indices_path=batch_indices_path,
        label_quantization=args.label_quantization, 
        dataset=args.dataset,   # use to distinguish imagenet21k
        transform=train_transform,
        use_rand_aug=args.use_rand_aug)
    args.total_num_batches = train_dataset.total_num_batches

    generator = torch.Generator()
    generator.manual_seed(args.fkd_seed)

    if args.prune_label is False:
        """Normal Setup"""
        sampler = torch.utils.data.RandomSampler(train_dataset, generator=generator)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
            num_workers=args.workers, pin_memory=True)
    else:
        """
        Sampler Setup
        Ensure that the same data is sampled for a given batch
        """
        # use the BatchSampler for flat structure
        sampler = utils_pruning.BatchSampler(train_dataset, generator=generator, batch_indices_path=batch_indices_path)
        print(f"using batch sampler with flat structure from {batch_indices_path}")
        
        # CRITICAL: We need a batch sampler that yields one original batch at a time
        # The current setup mixes indices from multiple batches, breaking the config alignment
        class PerBatchSampler:
            """Yields indices from one original batch at a time"""
            def __init__(self, base_sampler):
                self.base_sampler = base_sampler
                self.batch_list = None
            
            def set_batch_list(self, batch_list):
                self.batch_list = batch_list
                self.base_sampler.batch_list = None  # Don't let base sampler mix batches
            
            def __iter__(self):
                if self.batch_list is not None:
                    for batch_idx in self.batch_list:
                        # Set one batch at a time
                        self.base_sampler.set_batch(batch_idx)
                        # Yield all indices from this batch as a single batch
                        batch_indices = list(self.base_sampler)
                        if batch_indices:
                            yield batch_indices
            
            def __len__(self):
                return len(self.batch_list) if self.batch_list else 0
        
        per_batch_sampler = PerBatchSampler(sampler)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_sampler=per_batch_sampler,
            num_workers=args.workers, pin_memory=True)

    # load validation data
    val_dataset = MultiDatasetImageFolder(mode='val', dataset=args.dataset, root=args.val_dir, transform=val_transform)
    val_bs = int(args.batch_size/4) if args.batch_size > 64 else 64 # ensure val_bs is not too small
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_bs, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    print('load data successfully')

    return train_dataset, train_loader, val_loader

def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)

    # get training dynamics
    TD_logger = None

    os.makedirs(args.output_dir, exist_ok=True)
    cur_file = os.path.join(os.getcwd(), __file__)
    shutil.copy(cur_file, args.output_dir)

    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    global run
    if args.use_wandb:
        wandb_args = {"project": args.exp_name, "name": args.run_name}
        if args.tag_name is not None:
            wandb_args["tags"] = [args.tag_name]
        run = wandb.init(config={"tracking": False},
            settings=wandb.Settings(_disable_stats=True), 
            **wandb_args)
        wandb_hyperparam_log(args)
    else:
        run = None

    print_args_rich(args)

    if not torch.cuda.is_available():
        raise Exception("need gpu to train!")

    assert os.path.exists(args.train_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    _, train_loader, val_loader = load_dataset(args)

    # load student model
    print("=> loading student model '{}'".format(args.model))
    # imagenet21k-P is the pruned version of imagenet21k, containing 10,450 classes
    class_dict = {'imagenet1k': 1000, 'imagenet21k': 10450, 'tiny': 200}
    num_class = class_dict[args.dataset]
    args.num_class = num_class
    model = torchvision.models.__dict__[args.model](weights=None, num_classes=num_class)
    if args.dataset == 'tiny':
        # modifications for tiny imagenet
        # https://github.com/zeyuanyin/tiny-imagenet/tree/main?tab=readme-ov-file
        model.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        model.maxpool = nn.Identity()

    if args.train_epochs > 0:
        # scaling =  args.train_epochs / args.epochs 
        # args.adamw_lr = args.adamw_lr * scaling
        print(f"Scaling the learning rate to {args.adamw_lr}")
    else:
        args.train_epochs = args.epochs

    if args.dataset == 'tiny':
        model = model.cuda()    # a slight performance degradation is observed when using DataParallel
    else:
        model = nn.DataParallel(model).cuda()
    model.train()

    if args.ema:
        ema_model = EMA(model, beta=args.ema_decay)
    else:
        ema_model = None

    if args.sgd:
        optimizer = torch.optim.SGD(get_parameters(model),
                                    lr=args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(get_parameters(model),
                                      lr=args.adamw_lr,
                                      weight_decay=args.adamw_weight_decay)

    if args.cos == True:
        scheduler = LambdaLR(optimizer,
                             lambda step: 0.5 * (1. + math.cos(math.pi * step / args.train_epochs)) if step <= args.train_epochs else 0, last_epoch=-1)

        # handle special cases
        if args.smoothing_lr:
            # smoothing with strength 2
            scheduler = LambdaLR(optimizer,
                                lambda step: 0.5 * (1. + math.cos(math.pi * step / args.train_epochs / args.smoothing_lr_strength)) if step <= args.train_epochs else 0, last_epoch=-1)

        if args.sgd and (args.dataset == 'tiny'):
            # lr warm up for tiny-imagenet
            args.lr_warmup_epochs = 5
            args.lr_warmup_decay = 0.01
            args.lr_warmup_method = 'linear'
            args.lr_min = 0.0
            main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
            )
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
            )
    else:
        # default for SRe2L
        scheduler = LambdaLR(optimizer,
                            lambda step: (1.0-step/args.train_epochs) if step <= args.train_epochs else 0, last_epoch=-1)

    # Initialize temperature scheduler
    if args.temp_scheduler == 'step':
        def temp_scheduler(epoch):
            temperature = args.temperature * (args.temp_step_gamma ** (epoch // args.temp_step_size))
            return temperature if temperature > 2 else 2
    elif args.temp_scheduler == 'cosine':
        def temp_scheduler(epoch):
            return args.temp_min + 0.5 * (args.temperature - args.temp_min) * (1 + math.cos(math.pi * epoch / args.train_epochs))
    elif args.temp_scheduler == 'step_reverse':
        def temp_scheduler(epoch):
            if epoch >= args.train_epochs:
                epoch = args.train_epochs - 1   # no need to adjust for the last epoch
            return args.temperature * args.temp_step_gamma ** ((args.train_epochs - epoch - 1) // args.temp_step_size)

    elif args.temp_scheduler == 'cosine_reverse':
        def temp_scheduler(epoch):
            # reverse of cosine scheduler: starts at low temperature and increases
            # this reverses the entropy order from low to high
            # The cosine function is flipped to start at the minimum temperature and increase to the maximum temperature
            return args.temp_min + 0.5 * (args.temperature - args.temp_min) * (1 - math.cos(math.pi * epoch / args.train_epochs))
    else:
        def temp_scheduler(epoch):
            return args.temperature

    args.temp_scheduler_fn = temp_scheduler

    args.best_acc1=0
    args.optimizer = optimizer
    args.scheduler = scheduler
    args.train_loader = train_loader
    args.val_loader = val_loader
    
    # batch_to_epoch settings
    print("using batch_to_epoch approach")
    total_steps = args.train_epochs * args.train_loader.dataset.batch_num_per_epoch
    val_interval = args.val_interval*args.train_loader.dataset.batch_num_per_epoch
    save_interval = 1
    num_steps = args.train_loader.dataset.batch_num_per_epoch
    
    print("total_steps: {}, val_interval: {}, save_interval: {}, num_steps: {}".format(total_steps, val_interval, save_interval, num_steps))

    # === create pool for pruning ===
            
    # random pool
    pool = list(range(args.train_loader.dataset.total_num_batches))

    # assert int((1-args.prune_ratio)*total_steps) > len(pool), f"Not enough pool size, {int((1-args.prune_ratio)*total_steps)} < {len(pool)}, theoretical ratio: {1-args.prune_ratio} not met"
    print(f"Generated pool of size {len(pool)}")
    # ================================

    tracker = []
    stat_tracker = {}
    epoch_counter = 0
    for step in trange(0, total_steps, num_steps, desc="Training"):
        epoch_counter += 1
        if step % save_interval == 0:
            args.objs = AverageMeter()
            args.top1 = AverageMeter()
            args.top5 = AverageMeter()

        global logging_metrics
        logging_metrics = {}

        abs_step = step
        step_list = None

        # randomly sample epoch from pool
        if args.sample_metric == 'random':
            """
            for batch_to_epoch, which passes a list of batch indices
            """
            step_list = random.sample(pool, args.train_loader.dataset.batch_num_per_epoch)
            tracker += step_list
                
        elif args.sample_metric == 'order':
            step_list = list(range(step, step+args.train_loader.dataset.batch_num_per_epoch))
            # drop last batch
            for i, batch_idx in enumerate(step_list):
                if (batch_idx + 1) % args.train_loader.dataset.batch_num_per_epoch == 0:
                    step_list[i] = batch_idx - 1
        else:
            raise NotImplementedError

        current_epoch = abs_step//args.train_loader.dataset.batch_num_per_epoch+1

        print("Selected Batches: ", step_list[:10], " ...")
        train(model, args, step_list,
              TD_logger=TD_logger, abs_step=abs_step, interval=save_interval, ema_model=ema_model, current_epoch=current_epoch)


        if (abs_step % val_interval == 0) or (abs_step == total_steps - num_steps):
            top1, _ = validate(model if ema_model is None else ema_model, args, current_epoch)
        else:
            top1 = 0

        if args.use_wandb:
            logging_metrics.update({
                'epoch': current_epoch,
                'best_acc1': args.best_acc1,
                'temperature': args.temp_scheduler_fn(current_epoch)
            })
            wandb_log(logging_metrics, step=current_epoch)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = top1 > args.best_acc1
        args.best_acc1 = max(top1, args.best_acc1)
        if (abs_step % save_interval) == 0 or (abs_step == total_steps - num_steps):
            save_checkpoint({
                'epoch': current_epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': args.best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
            }, is_best, output_dir=args.output_dir)

    if run: # wandb save terminal log
        # track the histogram of epoch tracker
        run.log({"histogram": wandb.Histogram(np.array(tracker))}, step=0)
        dir_name = args.output_dir
        wandb_terminal_log(dir_name, args)

def adjust_bn_momentum(model, iters):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 1 / iters


def train(model, args, step, TD_logger=None, abs_step=None, interval=1, ema_model=None, current_epoch=None):
    t1 = time.time()
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_function = nn.CrossEntropyLoss()
    k_values = []
    running_sum_k = 0
    total_samples = 0
    # max_prob_list = []
    
    model.train()

    # set sampler to sample the images
    # Check if using batch_sampler (for prune_label=True) or regular sampler
    if hasattr(args.train_loader, 'batch_sampler') and args.train_loader.batch_sampler is not None:
        args.train_loader.batch_sampler.set_batch_list(step)
    else:
        args.train_loader.sampler.set_batch_list(step)

    for batch_idx, batch_data in enumerate(args.train_loader):

        if args.use_rand_aug:
            images, target, flip_status, coords_status, rand_params, idx = batch_data[0]
        else:
            images, target, flip_status, coords_status, idx = batch_data[0]

        label_indices = None
        if args.label_quantization:
            if args.label_quantization[:3] == 'MRS':
                mix_index, mix_lam, mix_bbox, soft_label, label_indices, sample_min = batch_data[1:]   # additional data from _MapDatasetFetcher
            else:   # normal MR
                mix_index, mix_lam, mix_bbox, soft_label, label_indices = batch_data[1:]   # additional data from _MapDatasetFetcher

        else:
            mix_index, mix_lam, mix_bbox, soft_label = batch_data[1:]   # additional data from _MapDatasetFetcher

        images = images.cuda()
        target = target.cuda()
        soft_label = soft_label.cuda().float()  # convert to float32

        # # ============ Shape Analysis =================
        # print("Shape of images: ", images.shape)
        # print("Shape of target: ", target.shape)
        # print("Shape of soft_label: ", soft_label.shape)
        # print("Shape of label_indices: ", label_indices.shape)
        # print("Mix index: ", mix_index.shape)
        # print("Mix bbox: ", mix_bbox)
        # # =============================================

        images, _, _, _ = mix_aug(images, args, mix_index, mix_lam, mix_bbox)

        optimizer = args.optimizer
        optimizer.zero_grad()
        assert args.batch_size % args.gradient_accumulation_steps == 0
        small_bs = args.batch_size // args.gradient_accumulation_steps

        # images.shape[0] is not equal to args.batch_size in the last batch, usually
        if batch_idx == len(args.train_loader) - 1:
            accum_step = math.ceil(images.shape[0] / small_bs)
        else:
            accum_step = args.gradient_accumulation_steps

        for accum_id in range(accum_step):
            partial_images = images[accum_id * small_bs: (accum_id + 1) * small_bs]
            partial_target = target[accum_id * small_bs: (accum_id + 1) * small_bs]
            partial_soft_label = soft_label[accum_id * small_bs: (accum_id + 1) * small_bs]

            # Get current temperature from scheduler if it exists
            current_temp = args.temp_scheduler_fn(current_epoch) if hasattr(args, 'temp_scheduler_fn') else args.temperature

            if args.label_smoothing > 0:
                num_classes = partial_soft_label.size(1)  # batch_size x num_classes
                uniform_distribution = torch.full((args.batch_size, num_classes), 1 / num_classes, device=partial_soft_label.device)
                # Adjust partial_soft_label with label smoothing
                partial_soft_label = (1 - args.label_smoothing) * partial_soft_label + args.label_smoothing * uniform_distribution

            output = model(partial_images)
            prec1, prec5 = accuracy(output, partial_target, topk=(1, 5))

            def apply_softmax(partial_soft_label):
                # softmax before quantization
                if args.dataset == 'tiny':
                    return F.log_softmax(partial_soft_label/current_temp, dim=1)
                else:
                    return F.softmax(partial_soft_label/current_temp, dim=1)

            # Apply label quantization before loss computation
            if args.label_quantization:
                # partial_soft_labels already store only topk predictions 
                # and label_indices are the class mapping
                
                label_indices = label_indices.to(partial_soft_label.device).long()

                if args.label_quantization[:3] == 'MRS':
                    sample_min = sample_min.to(partial_soft_label.device).float()
                    quantized_label = sample_min.expand(partial_soft_label.size(0), args.num_class).clone()
                    quantized_label.scatter_(1, label_indices, partial_soft_label)
                    partial_soft_label = apply_softmax(quantized_label)
                
                elif args.label_quantization[:2] == 'MS':
                    # NOTE: for MS implmentation, we store POST-softmax labels for a fixed temperature

                    # Create a full-dimensional tensor (1000 classes)
                    full_soft_label = torch.zeros(partial_soft_label.size(0), args.num_class, 
                                               device=partial_soft_label.device)
                    
                    # Vectorized operation: scatter the normalized values to their original positions
                    partial_soft_label = full_soft_label.scatter_(1, label_indices, partial_soft_label)

                elif args.label_quantization[:2] == 'MR':
                    # for mr, apply softmax to partial_soft_label first
                    partial_soft_label = apply_softmax(partial_soft_label)
                    
                    """no need for additional normalization, due to softmax on soley top-k values"""
                    
                    # Create a full-dimensional tensor (1000 classes)
                    full_soft_label = torch.zeros(partial_soft_label.size(0), args.num_class, 
                                               device=partial_soft_label.device)
                    
                    # Vectorized operation: scatter the normalized values to their original positions
                    partial_soft_label = full_soft_label.scatter_(1, label_indices, partial_soft_label)
                
                else:
                    raise NotImplementedError

                # max_prob, _ = torch.max(partial_soft_label, dim=1)
                # max_prob_list.extend(max_prob.cpu().tolist())
            else:
                partial_soft_label = apply_softmax(partial_soft_label)

            # determine student temperature
            if args.temp_stu_dynamic > 0:
                student_temperature = args.temp_stu_dynamic
            elif args.temp_stu > 0:
                student_temperature = args.temp_stu
            else: 
                student_temperature = current_temp
                
            output_log_softmax = F.log_softmax(output/student_temperature, dim=1)
            loss = nn.KLDivLoss(reduction='batchmean')(output_log_softmax, partial_soft_label)

            if args.dataset == 'tiny':
                assert accum_step == 1, "accum_step should be 1 for tiny-imagenet"
                # scale the loss by the temperature for tiny-imagenet
                loss = loss * (current_temp**2)

            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            n = partial_images.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if ema_model is not None:
                ema_model.update()

        optimizer.step()

    # Prepare metrics for WandB
    metrics = {
        "train/loss": objs.avg,
        "train/Top1": top1.avg,
        "train/Top5": top5.avg,
        "train/lr": args.scheduler.get_last_lr()[0],
        "train/epoch": current_epoch,
        "train/temperature": args.temp_scheduler_fn(current_epoch) if hasattr(args, 'temp_scheduler_fn') else args.temperature,
    }

    # determine dynamic temperature
    if args.temp_stu_dynamic > 0:
        # temperature ranges
        temperatures = np.linspace(0.01, 1, 100)
        loss_list = []
        for t in temperatures:
            temp_output = F.log_softmax(output/t, dim=1)
            temp_loss = nn.KLDivLoss(reduction='batchmean')(temp_output, partial_soft_label)
            loss_list.append(temp_loss.item())
        
        # find t that minimizes loss
        args.temp_stu_dynamic = temperatures[np.argmin(loss_list)]
        metrics["train/temp_stu_dynamic"] = args.temp_stu_dynamic
            
    # if len(max_prob_list) > 0:
    #     metrics["train/max_prob_crop"] = np.mean(max_prob_list)

    # Add K-related metrics if we collected any
    if len(k_values) > 0:
        k_array = np.array(k_values)
        k_metrics = {
            "train/k_mean": running_sum_k / total_samples,
            "train/k_min": np.min(k_array),
            "train/k_max": np.max(k_array),
            "train/k_std": np.std(k_array),
            "train/k_distribution": wandb.Histogram(k_array)
        }
        metrics.update(k_metrics)

        # Add K info to print statement
        printInfo = 'TRAIN Iter {}: lr = {:.6f},\ttemp = {:.2f},\tloss = {:.6f},\t'.format(
            "step" if not isinstance(step, list) else "...",  # Don't print the full list
            args.scheduler.get_last_lr()[0], args.temp_scheduler_fn(current_epoch) if hasattr(args, 'temp_scheduler_fn') else args.temperature, objs.avg) + \
                   'Top-1 err = {:.6f},\t'.format(100 - top1.avg) + \
                   'Top-5 err = {:.6f},\t'.format(100 - top5.avg) + \
                   'avg_k = {:.1f},\t'.format(k_metrics["train/k_mean"]) + \
                   'train_time = {:.6f}'.format((time.time() - t1))
    else:
        printInfo = 'TRAIN Iter {}: lr = {:.6f},\ttemp = {:.2f},\tloss = {:.6f},\t'.format(
            "step" if not isinstance(step, list) else "...",  # Don't print the full list
            args.scheduler.get_last_lr()[0], args.temp_scheduler_fn(current_epoch) if hasattr(args, 'temp_scheduler_fn') else args.temperature, objs.avg) + \
                   'Top-1 err = {:.6f},\t'.format(100 - top1.avg) + \
                   'Top-5 err = {:.6f},\t'.format(100 - top5.avg) + \
                   'train_time = {:.6f}'.format((time.time() - t1))

    if args.use_wandb:
        # Log training metrics at epoch level
        metrics.update({
            'epoch': current_epoch,
            'lr': args.scheduler.get_last_lr()[0],
        })
        wandb_log(metrics, step=current_epoch)
    print(printInfo)

def validate(model, args, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_function = nn.CrossEntropyLoss()

    model.eval()
    t1 = time.time()
    with torch.no_grad():
        for data, target in args.val_loader:
            target = target.type(torch.LongTensor)
            data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = loss_function(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    logInfo = 'TEST Iter {}: loss = {:.6f},\t'.format(epoch, objs.avg) + \
              'Top-1 err = {:.6f},\t'.format(100 - top1.avg) + \
              'Top-5 err = {:.6f},\t'.format(100 - top5.avg) + \
              'val_time = {:.6f}'.format(time.time() - t1)
    print(logInfo)

    metrics = {
        'val/loss': objs.avg,
        'val/top1': top1.avg,
        'val/top5': top5.avg,
        'val/epoch': epoch,
    }

    if args.use_wandb:
        # Log validation metrics at epoch level
        wandb_log(metrics, step=epoch)

    return top1.avg, None

def save_checkpoint(state, is_best, output_dir=None,epoch=None):
    if epoch is None:
        path = output_dir + '/' + 'checkpoint.pth.tar'
    else:
        path = output_dir + f'/checkpoint_{epoch}.pth.tar'
    torch.save(state, path)

    if is_best:
        path_best = output_dir + '/' + 'model_best.pth.tar'
        shutil.copyfile(path, path_best)

def get_adaptive_topk(soft_label, threshold=0.95):
    """
    Get adaptive top-k where cumulative sum exceeds threshold while minimizing K
    Args:
        soft_label: tensor of shape (batch_size, num_classes) containing probabilities
        threshold: float between 0 and 1, target cumulative probability
    Returns:
        topk_values: tensor of top-k values for each sample
        topk_indices: tensor of top-k indices for each sample
    """
    # Sort values in descending order
    sorted_probs, sorted_indices = torch.sort(soft_label, dim=1, descending=True)
    
    # Get cumulative sum
    cumsum = torch.cumsum(sorted_probs, dim=1)
    
    # Find minimum k where cumsum >= threshold
    # mask will be True for all positions where cumsum >= threshold
    mask = cumsum >= threshold
    
    # Get the first True position for each row (minimum k that exceeds threshold)
    # Adding 1 because we want to include the position that exceeded threshold
    k_positions = torch.argmax(mask.float(), dim=1) + 1
    
    # Get max k for creating output tensors
    max_k = torch.max(k_positions).item()
    
    # Initialize output tensors
    batch_size = soft_label.size(0)
    device = soft_label.device
    
    topk_values = torch.zeros((batch_size, max_k), device=device)
    topk_indices = torch.zeros((batch_size, max_k), device=device, dtype=torch.long)
    
    # Fill in values and indices for each sample
    for i in range(batch_size):
        k = k_positions[i]
        topk_values[i, :k] = sorted_probs[i, :k]
        topk_indices[i, :k] = sorted_indices[i, :k]
    
    return NamedTuple('TopK', [('values', torch.Tensor), ('indices', torch.Tensor)])(topk_values, topk_indices)

if __name__ == "__main__":
    # import multiprocessing as mp
    # mp.set_start_method('spawn')
    main()
