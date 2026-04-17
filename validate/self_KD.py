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
from utils import print_args_rich
from utils import AverageMeter, accuracy, get_parameters
from utils_fkd import mix_aug
from imagenet_ipc import ImageFolderIPC

# import the official resnet implementation
sys.path.append(os.path.join(os.path.dirname(__file__), '../pytorch-self-distillation-final'))
from resnet import resnet18, resnet34, resnet50, resnet101, resnet152

def get_args():
    parser = argparse.ArgumentParser("Self-KD Training on ImageNet-1K")
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
                        default='resnet18', help='model name')

    parser.add_argument('-T', '--temperature', type=float,
                        default=3.0, help='temperature for distillation loss')
    parser.add_argument('--loss-coefficient', type=float,
                        default=0.3, help='weight for distillation loss')
    parser.add_argument('--feature-loss-coefficient', type=float,
                        default=0.03, help='weight for feature distillation loss')
    parser.add_argument('--use-feature-loss', action='store_true',
                        help='whether to use feature distillation loss')
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
    parser.add_argument('--label-quantization', type=str, default=None, help='label quantization method')

    args = parser.parse_args()
    args.mode = 'fkd_save' # set for `mix_aug`
    return args

def CrossEntropy(outputs, targets, temperature):
    log_softmax_outputs = F.log_softmax(outputs/temperature, dim=1)
    softmax_targets = F.softmax(targets/temperature, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()

def main():
    args = get_args()

    wandb.init(config={"tracking": False},
               settings=wandb.Settings(_disable_stats=True))

    print_args_rich(args)

    if not torch.cuda.is_available():
        raise Exception("need gpu to train!")

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

    print("=> loading model '{}'".format(args.model))
    if args.model == "resnet18":
        model = resnet18(num_classes=1000)
    elif args.model == "resnet34":
        model = resnet34(num_classes=1000)
    elif args.model == "resnet50":
        model = resnet50(num_classes=1000)
    elif args.model == "resnet101":
        model = resnet101(num_classes=1000)
    elif args.model == "resnet152":
        model = resnet152(num_classes=1000)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    model = nn.DataParallel(model).cuda()
    model.train()

    if args.sgd_setting:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        assert args.cos, "CosineAnnealing Scheduler is used in SGD setting"
    else:  # default to use AdamW
        optimizer = torch.optim.AdamW(get_parameters(model),
                                    lr=args.adamw_lr,
                                    weight_decay=args.adamw_weight_decay)

    if args.cos:
        scheduler = LambdaLR(optimizer,
                           lambda step: 0.5 * (1. + math.cos(math.pi * step / args.epochs)) if step <= args.epochs else 0, last_epoch=-1)
    else:
        scheduler = LambdaLR(optimizer,
                           lambda step: (1.0-step/args.epochs) if step <= args.epochs else 0, last_epoch=-1)

    args.best_acc1 = 0
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

def train(model, args, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    optimizer = args.optimizer
    scheduler = args.scheduler
    loss_function = nn.CrossEntropyLoss()

    model.train()
    t1 = time.time()
    max_prob_list = []

    for batch_idx, (data, target) in enumerate(args.train_loader):
        target = target.type(torch.LongTensor)
        data, target = data.cuda(), target.cuda()

        images, _, _, _ = mix_aug(data, args)
        outputs = model(images)
        
        # compute loss
        loss = torch.FloatTensor([0.]).cuda()
        
        # loss for deepest classifier (teacher)
        loss += loss_function(outputs, target)
        
        teacher_output = outputs.detach()
        teacher_feature = model.module.features(images).detach()
        
        # losses for shallower classifiers (students)
        for i in range(1, len(model.module.features)):
            # cross entropy with true labels
            loss += loss_function(model.module.features[i](images), target) * (1 - args.loss_coefficient)
            
            # KL divergence with teacher's predictions
            loss += CrossEntropy(model.module.features[i](images), teacher_output, args.temperature) * args.loss_coefficient
            
            # feature distillation (except for shallowest classifier)
            if i != len(model.module.features)-1 and args.use_feature_loss:  # skip the shallowest classifier
                if not hasattr(model.module, 'adaptation_layers'):
                    # initialize adaptation layers
                    layer_list = []
                    teacher_feature_size = teacher_feature.size(1)
                    for j in range(1, len(model.module.features)):
                        student_feature_size = model.module.features[j](images).size(1)
                        layer_list.append(nn.Linear(student_feature_size, teacher_feature_size).cuda())
                    model.module.adaptation_layers = nn.ModuleList(layer_list)
                
                # compute feature distillation loss
                adapted_feature = model.module.adaptation_layers[i-1](model.module.features[i](images))
                loss += torch.dist(adapted_feature, teacher_feature) * args.feature_loss_coefficient

        # compute accuracy for deepest classifier
        prec1, prec5 = accuracy(outputs, target, topk=(1, 5))
        
        n = images.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        
        # record max probability
        max_prob, _ = torch.max(F.softmax(outputs, dim=1), dim=1)
        max_prob_list.append(max_prob.mean().item())

        if batch_idx == 0:
            optimizer.zero_grad()

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or batch_idx == len(args.train_loader) - 1:
            optimizer.step()
            optimizer.zero_grad()

    metrics = {
        "train/loss": objs.avg,
        "train/Top1": top1.avg,
        "train/Top5": top5.avg,
        "train/lr": scheduler.get_last_lr()[0],
        "train/epoch": epoch,
        "train/max_prob_crop": np.mean(max_prob_list)
    }

    printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(epoch, scheduler.get_last_lr()[0], objs.avg) + \
               'Top-1 err = {:.6f},\t'.format(100 - top1.avg) + \
               'Top-5 err = {:.6f},\t'.format(100 - top5.avg) + \
               'train_time = {:.6f}'.format((time.time() - t1))

    wandb.log(metrics, step=epoch)
    print(printInfo)
    scheduler.step()
    t1 = time.time()
    return np.mean(max_prob_list)

def validate(model, args, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_function = nn.CrossEntropyLoss()

    model.eval()
    t1 = time.time()

    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for data, target in args.val_loader:
            target = target.type(torch.LongTensor)
            data, target = data.cuda(), target.cuda()

            outputs = model(data)
            loss = loss_function(outputs, target)

            prec1, prec5 = accuracy(outputs, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            _, predicted = torch.max(outputs, 1)
            correct = (predicted == target).squeeze()
            for i in range(n):
                label = target[i].item()
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    per_class_accuracy = {label: class_correct[label] / class_total[label] 
                          for label in class_total}

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

    wandb.log(metrics, step=epoch)

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
