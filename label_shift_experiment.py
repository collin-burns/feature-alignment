import argparse
import os
import os.path
import time

import numpy as np
import matplotlib
import copy

matplotlib.use('agg')
import matplotlib.pyplot as plt
from itertools import product
import torch
import torchvision
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.datasets as dset
import torch.distributions as distributions
import torchvision.transforms as trn
import torch.nn as nn
import torch.backends.cudnn as cudnn
from models.wrn import WideResNet
from utils import load_model, get_dataset, get_labels
from evaluate_all import get_acc_conf_ce, get_num_norm, bn_on, update_bn_stats, get_results_c, get_individual_results

normalize = trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
imagenet_transform = trn.Compose([
    trn.Resize(256),
    trn.CenterCrop(224),
    trn.ToTensor(),
    normalize,
])

def select_classes_dataset(dataset, keep_classes, num_classes=10):
    labels = get_labels(torch.utils.data.DataLoader(dataset, batch_size=1028)).squeeze()
    idxs = torch.zeros(len(labels)).bool()
    for k in range(num_classes):
        if k in keep_classes:
            idxs[labels == k] = True
    indices = list(np.argwhere(idxs.flatten()).flatten())
    return torch.utils.data.Subset(dataset, indices=indices)

def main(args):
    device = "cuda" if args.ngpu > 0 else "cpu"
    save_dir = os.path.join("final_results", "label_shift", args.dataset)
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    test_dataset, test_transform, num_classes, size = get_dataset(args)

    if args.dataset == "CIFAR-10-C":
        train_transform = trn.Compose([trn.ToPILImage(), trn.RandomHorizontalFlip(), trn.RandomCrop(size, padding=4), test_transform])
    elif args.dataset in ["CIFAR-10", "TIN-C", "TIN"]:
        train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(size, padding=4), test_transform])
    else:
        train_transform = trn.Compose([trn.RandomResizedCrop(224), trn.RandomHorizontalFlip(), trn.ToTensor(), normalize])

    assert args.norm == "batch_norm"
    model = load_model(args, device, num_classes)
    num_bn = get_num_norm(model, nn.BatchNorm2d)

    if args.dataset in ["CIFAR-10", "CIFAR-10-C"]:
        keep_class_sizes = [1, 3, 6, 10]
    elif args.dataset in ["TIN-C", "TIN"]:
        keep_class_sizes = [20, 60, 120, 200]
    else:
        raise NotImplementedError

    ks = [0, 1, 4, 16]  # number of BN layers at end to exclude
    all_idxs = []
    for k in ks:
        idxs = np.ones(num_bn).astype(bool)
        if k == 0:
            all_idxs.append(idxs)
        else:
            if args.exclude_initial:
                idxs[:k] = False 
            else:
                idxs[-k:] = False  # don't want to do [-0:]
            all_idxs.append(idxs)

    for keep_class_size in keep_class_sizes:
        keep_classes = [i for i in range(keep_class_size)]

        for k, idxs in zip(ks, all_idxs):
            if args.dataset in ["CIFAR-10-C", "CIFAR-100-C", "TIN-C", "ImageNet-C"]:
                print("Starting BN update with aug")
                tr_bn_c_results = get_results_c(args, model, device, update_bn=True, val_transform=train_transform,
                                                idxs=idxs, keep_classes=keep_classes)
                if args.exclude_initial:
                    np.save(os.path.join(save_dir, "tr_bn_exclude_initial_{:02d}_keep_{:02d}.npy".format(k, keep_class_size)),
                            tr_bn_c_results)
                else:
                    np.save(os.path.join(save_dir, "tr_bn_exclude_{:02d}_keep_{:02d}.npy".format(k, keep_class_size)), tr_bn_c_results)
            else:
                tr_bn_results = get_individual_results(args, model, device, update_bn=True,
                                                       val_transform=train_transform, idxs=idxs,
                                                       keep_classes=keep_classes)
                if args.exclude_initial:
                    np.save(os.path.join(save_dir, "tr_bn_exclude_initial_{:02d}_keep_{:02d}.npy".format(k, keep_class_size)),
                            tr_bn_results)
                else:
                    np.save(os.path.join(save_dir, "tr_bn_exclude_{:02d}_keep_{:02d}.npy".format(k, keep_class_size)), tr_bn_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CIFAR-10-C")
    parser.add_argument("--norm", type=str, default="batch_norm")
    parser.add_argument("--ngroups", type=int, default=4, help="Number of groups if GroupNorm is used.")
    parser.add_argument("--no_affine", action="store_true", help="Whether to not use affine parameters for normalization.")
    parser.add_argument("--exclude_initial", action="store_true")
    parser.add_argument("--model", type=str, default="WideResNet")
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("--layers", default=40, type=int, help="total number of layers")
    parser.add_argument("--widen-factor", default=2, type=int, help="widen factor")
    parser.add_argument("--droprate", default=0.3, type=float, help="dropout probability")
    parser.add_argument("--ngpu", type=int, default=1, help="0 = CPU.")
    parser.add_argument("--load_dir", "-d", type=str, default="./snapshots", required=True)
    parser.add_argument("--root", type=str, default="./data", required=True)
    args = parser.parse_args()
    main(args)

