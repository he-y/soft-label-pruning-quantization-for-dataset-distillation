import os
import sys
import math
import time
import shutil
import argparse
import numpy as np
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LambdaLR

from collections import defaultdict
import matplotlib.pyplot as plt
from utils import print_args_rich

from imagenet_ipc import ImageFolderIPC
from utils import AverageMeter, accuracy, get_parameters
from utils_fkd import mix_aug

from typing import NamedTuple

def get_args():
    parser = argparse.ArgumentParser("KD Training on ImageNet-1K")
    parser.add_argument('--batch-size', type=int,
                        default=1024, help='batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int,
                        default=1, help='gradient accumulation steps for small gpu memory')
    parser.add_argument('--start-epoch', type=int,
                        default=0, help='start epoch')
    parser.add_argument('--epochs', type=int, default=300, help='total epoch')
    parser.add_argument('-j', '--workers', default=16, type=int,
                        help='number of data loading workers')

    parser.add_argument('--train-dir', type=str, default=None,
                        help='path to training dataset')
    parser.add_argument('--val-dir', type=str,
                        default='/path/to/imagenet/val', help='path to validation dataset')
    parser.add_argument('--output-dir', type=str,
                        default=None, help='path to output dir')

    parser.add_argument('--cos', default=False,
                        action='store_true', help='cosine lr scheduler')
    parser.add_argument('--adamw-lr', type=float,
                        default=0.001, help='adamw learning rate')
    parser.add_argument('--adamw-weight-decay', type=float,
                        default=0.01, help='adamw weight decay')


    parser.add_argument('--model', type=str,
                        default='resnet18', help='student model name')
    parser.add_argument('--teacher-model', type=str, default=None,
                        help='teacher model name')

    parser.add_argument('-T', '--temperature', type=float,
                        default=3.0, help='temperature for distillation loss')
    # parser.add_argument('--wandb-project', type=str,
    #                     default='Temperature', help='wandb project name')
    # parser.add_argument('--wandb-api-key', type=str,
    #                     default=None, help='wandb api key')
    parser.add_argument('--mix-type', default=None, type=str,
                        choices=['mixup', 'cutmix', None], help='mixup or cutmix or None')
    parser.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--IPC', default=50, type=int, help='number of images per class')
    
    parser.add_argument('--hard-label', default=False, action='store_true', help='use hard label')
    parser.add_argument('--sgd-setting', default=False, action='store_true', help='using sgd evaluation settting (lr=0.1, scheduler=cos)')

    parser.add_argument('--hf-cache-dir', type=str, default='./.hf_cache', help='cache dir for huggingface dataset')
    # ablation: comparison with label compression (FKD)
    """
    choices:
    1. 'hard': hard label quantization (one-hot)
    2. 'smooth': smooth label quantization ()
    3. 'MS-[topk]': marginal smoothing with top-k (e.g., MS-10)
    4. 'MR-[topk]': marginal re-norm with top-k (e.g., MR-10)
    """
    parser.add_argument('--label-quantization', type=str, default=None, help='label quantization method')

    args = parser.parse_args()
    args.mode = 'fkd_save' # set for `mix_aug`
    return args

def main():
    args = get_args()

    # wandb.login(key=args.wandb_api_key)
    wandb.init(config={"tracking": False},
               settings=wandb.Settings(_disable_stats=True))

    print_args_rich(args)

    # print('=> args.output_dir', args.output_dir)

    if not torch.cuda.is_available():
        raise Exception("need gpu to train!")


    # assert os.path.exists(args.train_dir), print(f"train_dir not found: {args.train_dir}")
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)

    # Data loading
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])


    if "he-y" in args.train_dir:  # use huggingface dataset
        from datasets import load_dataset
        from imagenet_ipc import HFDatasetAdapter
        dataset_hf = load_dataset(args.train_dir, cache_dir=args.hf_cache_dir)
        train_dataset = HFDatasetAdapter(dataset_hf, transform=train_transforms)
        print(f"=> Load data from Huggingface: total images = {len(train_dataset)}, choose images = {args.IPC}")
    else:   # use local dataset
        assert os.path.exists(args.train_dir)
        train_dataset = ImageFolderIPC(
            args.train_dir,
            transform=train_transforms,
            image_number=args.IPC
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # load validation data
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.val_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=int(args.batch_size/4), shuffle=False,
        num_workers=args.workers, pin_memory=True)
    print('load data successfully')


    # load student model
    print("=> loading student model '{}'".format(args.model))
    model = torchvision.models.__dict__[args.model](weights=None)
    model = nn.DataParallel(model).cuda()
    model.train()

    if not args.hard_label:
        # load teacher model
        print("=> loading teacher model '{}'".format(args.teacher_model))
        teacher_model = torchvision.models.__dict__[args.teacher_model](pretrained=True)
        teacher_model = nn.DataParallel(teacher_model).cuda()
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        args.teacher_model = teacher_model

    if args.sgd_setting:
        # pre-defined as pruning settings
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        assert args.cos, "CosineAnnealing Scheduler is used in SGD setting"
    else:  # default to use AdamW
        optimizer = torch.optim.AdamW(get_parameters(model),
                                        lr=args.adamw_lr,
                                        weight_decay=args.adamw_weight_decay)

    if args.cos == True:
        scheduler = LambdaLR(optimizer,
                             lambda step: 0.5 * (1. + math.cos(math.pi * step / args.epochs)) if step <= args.epochs else 0, last_epoch=-1)
    else:
        scheduler = LambdaLR(optimizer,
                             lambda step: (1.0-step/args.epochs) if step <= args.epochs else 0, last_epoch=-1)


    args.best_acc1=0
    args.optimizer = optimizer
    args.scheduler = scheduler
    args.train_loader = train_loader
    args.val_loader = val_loader
    

    max_prob_crop_list = []
    for epoch in range(args.start_epoch, args.epochs):
        print(f"\nEpoch: {epoch}")

        global wandb_metrics
        wandb_metrics = {}

        max_prob_crop = train(model, args, epoch)
        max_prob_crop_list.append(max_prob_crop)

        if epoch % 50 == 0 or epoch == args.epochs - 1:
            top1, _ = validate(model, args, epoch)
        else:
            top1 = 0

        wandb.log(wandb_metrics, step=epoch)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = top1 > args.best_acc1
        args.best_acc1 = max(top1, args.best_acc1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': args.best_acc1,
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict(),
        }, is_best, output_dir=args.output_dir)
    wandb.log({'max_prob_crop.avg': np.mean(max_prob_crop_list)}, step=epoch)

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

def train(model, args, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # K tracking variables
    k_values = []
    running_sum_k = 0
    total_samples = 0

    optimizer = args.optimizer
    scheduler = args.scheduler
    loss_function_kl = nn.KLDivLoss(reduction='batchmean')
    loss_function = nn.CrossEntropyLoss()

    model.train()
    if not args.hard_label:
        args.teacher_model.eval()
    t1 = time.time()
    max_prob_list = []

    for batch_idx, (data, target) in enumerate(args.train_loader):
        target = target.type(torch.LongTensor)
        data, target = data.cuda(), target.cuda()

        images, _, _, _ = mix_aug(data, args)
        output = model(images)
        
        if not args.hard_label:
            soft_label = args.teacher_model(images).detach()
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            output = F.log_softmax(output/args.temperature, dim=1)
            soft_label = F.softmax(soft_label/args.temperature, dim=1)
            
            if args.label_quantization:
                soft_label = soft_label.cuda()

                if args.label_quantization == 'hard':
                    soft_label = F.one_hot(torch.argmax(soft_label, dim=1), 
                                         num_classes=soft_label.size(1)).float()
                else:
                    # extract top-k labels or adaptive threshold
                    topk = args.label_quantization.split('-')[1]
                    adaptive = True if topk[0] == '0' else False # MS-10 or MS-0.9
                    if adaptive: 
                        topk = float(topk)
                        topk_labels = get_adaptive_topk(soft_label, topk)
                        # Track K statistics
                        actual_k = torch.count_nonzero(topk_labels.values, dim=1)
                        k_values.extend(actual_k.cpu().tolist())
                        running_sum_k += actual_k.sum().item()
                        total_samples += actual_k.size(0)
                    else:
                        topk = int(topk)
                        topk_labels = torch.topk(soft_label, topk, dim=1)

                    if args.label_quantization[:2] == 'MS':
                        smoothed_values = (torch.ones(soft_label.size(0),1, device=soft_label.device) - 
                                        torch.sum(topk_labels.values, dim=1, keepdim=True)) / (soft_label.size(1) - topk)

                        quantized_label = smoothed_values.expand_as(soft_label).clone()
                        quantized_label.scatter_(1, topk_labels.indices, topk_labels.values)
                        
                        soft_label = quantized_label

                    elif args.label_quantization[:3] == 'MRS':
                        # combination of MR and MS
                        # normalize the top-k and smoothing the remaining (instead of setting to zero)
                        
                        # Step 1: Normalize the top-k values (MR part)
                        normalized_topk_values = F.normalize(topk_labels.values, p=1, dim=1)
                        
                        # Step 2: Calculate what proportion of the probability mass is allocated to top-k
                        alpha = torch.sum(normalized_topk_values, dim=1, keepdim=True)
                        
                        # Step 3: Calculate smoothing value for non-top-k (MS part)
                        # Remaining probability mass (1-alpha) distributed equally
                        smoothed_values = (torch.ones(soft_label.size(0),1, device=soft_label.device) - alpha) / (soft_label.size(1) - topk)
                        
                        # Step 4: Combine
                        quantized_label = smoothed_values.expand_as(soft_label).clone()
                        quantized_label.scatter_(1, topk_labels.indices, normalized_topk_values)
                        
                        soft_label = quantized_label

                    elif args.label_quantization[:2] == 'MR':
                        quantized_label = torch.zeros_like(soft_label)
                        quantized_label.scatter_(1, topk_labels.indices, topk_labels.values)
                        
                        quantized_label = F.normalize(quantized_label, p=1, dim=1)
                        
                        soft_label = quantized_label
                    else:
                        raise NotImplementedError

            max_prob, _ = torch.max(soft_label, dim=1)
            max_prob_list.append(max_prob.mean().item())

            loss = loss_function_kl(output, soft_label)
        else:
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            loss = loss_function(output, target)

        n = images.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if batch_idx == 0:
            optimizer.zero_grad()

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or batch_idx == len(args.train_loader) - 1:
            optimizer.step()
            optimizer.zero_grad()

    # Prepare metrics for WandB
    metrics = {
        "train/loss": objs.avg,
        "train/Top1": top1.avg,
        "train/Top5": top5.avg,
        "train/lr": scheduler.get_last_lr()[0],
        "train/epoch": epoch,
        "train/max_prob_crop": np.mean(max_prob_list)
    }

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
        printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(epoch, scheduler.get_last_lr()[0], objs.avg) + \
                   'Top-1 err = {:.6f},\t'.format(100 - top1.avg) + \
                   'Top-5 err = {:.6f},\t'.format(100 - top5.avg) + \
                   'avg_k = {:.1f},\t'.format(k_metrics["train/k_mean"]) + \
                   'train_time = {:.6f}'.format((time.time() - t1))
    else:
        # Original print statement
        printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(epoch, scheduler.get_last_lr()[0], objs.avg) + \
                   'Top-1 err = {:.6f},\t'.format(100 - top1.avg) + \
                   'Top-5 err = {:.6f},\t'.format(100 - top5.avg) + \
                   'train_time = {:.6f}'.format((time.time() - t1))

    wandb.log(metrics, step=epoch)
    print(printInfo)
    t1 = time.time()
    return np.mean(max_prob_list)

def validate(model, args, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_function = nn.CrossEntropyLoss()

    model.eval()
    t1 = time.time()

    # Initialize per-class accuracy tracking
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

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

            # Calculate per-class accuracy
            _, predicted = torch.max(output, 1)
            correct = (predicted == target).squeeze()
            for i in range(n):
                label = target[i].item()
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    # Calculate per-class accuracy
    per_class_accuracy = {label: class_correct[label] / class_total[label] 
                          for label in class_total}

    logInfo = 'TEST Iter {}: loss = {:.6f},\t'.format(epoch, objs.avg) + \
              'Top-1 err = {:.6f},\t'.format(100 - top1.avg) + \
              'Top-5 err = {:.6f},\t'.format(100 - top5.avg) + \
              'val_time = {:.6f}'.format(time.time() - t1)
    print(logInfo)

    # # Visualize per-class accuracy
    # plt.figure(figsize=(20, 10))
    # plt.bar(per_class_accuracy.keys(), per_class_accuracy.values())
    # # plot average accuracy
    # plt.axhline(y=top1.avg/100, color='r', linestyle='--', label='average accuracy', linewidth=2)
    # plt.title('Per-Class Accuracy')
    # plt.xlabel('Class')
    # plt.ylabel('Accuracy')
    # plt.savefig('per_class_accuracy.png')
    # plt.close()

    metrics = {
        'val/loss': objs.avg,
        'val/top1': top1.avg,
        'val/top5': top5.avg,
        'val/epoch': epoch,
    }

    # # Add per-class accuracy to wandb metrics
    # for label, acc in per_class_accuracy.items():
    #     metrics[f'val/class_{label}_accuracy'] = acc

    wandb.log(metrics, step=epoch)

    # # Upload the per-class accuracy plot to wandb
    # wandb.log({"per_class_accuracy_plot": wandb.Image('per_class_accuracy.png')})

    return top1.avg, per_class_accuracy


def save_checkpoint(state, is_best, output_dir=None,epoch=None):
    print('==> Do not save checkpoint to save space')
    return
    if epoch is None:
        path = output_dir + '/' + 'checkpoint.pth.tar'
    else:
        path = output_dir + f'/checkpoint_{epoch}.pth.tar'
    torch.save(state, path)

    if is_best:
        path_best = output_dir + '/' + 'model_best.pth.tar'
        shutil.copyfile(path, path_best)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn')
    main()
    wandb.finish()
