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
from utils import calib_err, load_model, load_corrupted_dataset, get_dataset, get_model_bn_stats
from utils import select_classes_dataset, get_num_classes

test_corruptions = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression"]

def update_bn_stats(orig_model, bn_loader, device, idxs=None):
    model = copy.deepcopy(orig_model)
    if bn_loader is not None:
        model = bn_on(model, idxs)
        with torch.no_grad():
            for x, y in bn_loader:
                x = x.to(device)
                _ = model(x)
    model.eval()
    return model


def get_acc_conf_ce(model, device, eval_loader):
    model.eval()
    cors = []
    confs = []
    for x, y in eval_loader:
        x = x.to(device)
        output = model(x)

        pred = output.detach().cpu().numpy().argmax(1)
        cor = (pred == y.detach().cpu().numpy())
        conf = torch.softmax(output, dim=1).detach().cpu().numpy().max(1)

        cors.append(cor)
        confs.append(conf)

    cors = np.concatenate(cors)
    confs = np.concatenate(confs)
    ce = calib_err(confs, cors)
    acc = cors.mean()
    conf = confs.mean()
    return acc, conf, ce

def get_results_c(args, orig_model, device, update_bn=False, val_transform=None, idxs=None, keep_classes=None, save_stats=True):
    all_accs, all_confs, all_ces = [], [], []
    stats = {}
    severities = [1, 2, 3, 4, 5]
    num_classes = get_num_classes(args.dataset)
    for tr in test_corruptions:
        inner_accs = []
        inner_confs = []
        inner_ces = []
        for sev in severities:
            val_dataset = load_corrupted_dataset(args.dataset, args.root, tr, sev)
            if keep_classes is not None:
                val_dataset = select_classes_dataset(val_dataset, keep_classes, num_classes)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True)

            if val_transform is not None:
                bn_val_dataset = load_corrupted_dataset(args.dataset, args.root, tr, sev, transform=val_transform)
                if keep_classes is not None:
                    bn_val_dataset = select_classes_dataset(bn_val_dataset, keep_classes, num_classes)
                bn_val_loader = torch.utils.data.DataLoader(bn_val_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True)
            else:
                bn_val_loader = val_loader

            if args.norm == "batch_norm" and update_bn:
                model = update_bn_stats(orig_model, bn_val_loader, device, idxs)
            else:
                model = orig_model

            acc, conf, ce = get_acc_conf_ce(model, device, val_loader)
            print("{} {} acc {:.3f} conf {:.3f} ce {:.3f}".format(tr, sev, acc, conf, ce))
            means, vars = get_model_bn_stats(model)
            stats[(tr, sev)] = (means, vars)

            inner_accs.append(acc)
            inner_confs.append(conf)
            inner_ces.append(ce)
        all_accs.append(inner_accs)
        all_confs.append(inner_confs)
        all_ces.append(inner_ces)
    print("Average acc: {:.3f}".format(np.mean(all_accs)))
    if save_stats:
        results = {"accs": all_accs, "confs": all_confs, "cerrs": all_ces, "stats": stats}
    else:
        results = {"accs": all_accs, "confs": all_confs, "cerrs": all_ces}
    return results

normalize = trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
imagenet_transform = trn.Compose([
    trn.Resize(256),
    trn.CenterCrop(224),
    trn.ToTensor(),
    normalize,
])

def get_individual_results(args, orig_model, device, update_bn=False, val_transform=None, idxs=None, save_stats=True,
                           keep_classes=None):
    num_classes = get_num_classes(args.dataset)
    if args.dataset in ["CIFAR-10", "CIFAR-100", "TIN", "ImageNet"]:
        val_dataset, _, _, _ = get_dataset(args)
    elif args.dataset == "ImageNet-v2":
        val_dataset = dset.ImageFolder(os.path.join(args.root, "imagenet-matched-frequency-format-val"),
                                       transform=imagenet_transform)
    elif args.dataset == "StylizedImageNet":
        val_dataset = dset.ImageFolder(os.path.join(args.root, "stylized_imagenet/val"),
                                       transform=imagenet_transform)
    else:
        raise NotImplementedError
    if keep_classes is not None:
        val_dataset = select_classes_dataset(val_dataset, keep_classes, num_classes)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True)

    if val_transform is not None:
        # then get a separate loader with data aug for updating the BN statistics
        if args.dataset in ["CIFAR-10", "CIFAR-100", "TIN", "ImageNet"]:
            bn_val_dataset, _, _, _ = get_dataset(args, val_transform)
        elif args.dataset == "ImageNet-v2":
            bn_val_dataset = dset.ImageFolder(os.path.join(args.root, "imagenetv2-matched-frequency"),
                                              transform=val_transform)
        elif args.dataset == "StylizedImageNet":
            bn_val_dataset = dset.ImageFolder(os.path.join(args.root, "stylized_imagenet/val"),
                                              transform=val_transform)
        if keep_classes is not None:
            bn_val_dataset = select_classes_dataset(bn_val_dataset, keep_classes, num_classes)
        bn_val_loader = torch.utils.data.DataLoader(bn_val_dataset, batch_size=args.batch_size, pin_memory=True,
                                                    shuffle=True)
    else:
        bn_val_loader = val_loader

    if args.norm == "batch_norm" and update_bn:
        model = update_bn_stats(orig_model, bn_val_loader, device, idxs)
    else:
        model = orig_model

    acc, conf, ce = get_acc_conf_ce(model, device, val_loader)
    print("acc {:.3f} conf {:.3f} ce {:.3f}".format(acc, conf, ce))
    if save_stats:
        stats = get_model_bn_stats(model)
        results = {"accs": acc, "confs": conf, "cerrs": ce, "stats": stats}
    else:
        results = {"accs": acc, "confs": conf, "cerrs": ce}
    return results

def get_num_norm(model, norm):
    idx = [0]
    def g(m, idx=idx):
        if type(m) == norm:
            idx[0] += 1
    model.apply(g)
    return idx[0]


def bn_on(model, idxs=None):
    """
    idxs[l] indicates whether to turn on for the l'th BN layer.
    """
    model.eval()

    idx = [0]  # list so that it's modified
    if idxs is None:
        num_bn = get_num_norm(model, nn.BatchNorm2d)
        idxs = np.ones(num_bn).astype(bool)

    def f(m, idx=idx, idxs=idxs):
        if type(m) == nn.BatchNorm2d:
            if idxs[idx[0]]:
                m.train(True)
            idx[0] += 1

    model.apply(f)
    return model

def main(args):
    device = "cuda" if args.ngpu > 0 else "cpu"
    save_dir = os.path.join("final_results", "robustness_comparison", args.dataset)
    if args.model != "WideResNet":
        save_dir = os.path.join(save_dir, args.model)
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    test_dataset, test_transform, num_classes, size = get_dataset(args)

    if args.dataset == "CIFAR-10-C":
        train_transform = trn.Compose([trn.ToPILImage(), trn.RandomHorizontalFlip(), trn.RandomCrop(size, padding=4), test_transform])
    elif args.dataset in ["CIFAR-10", "TIN-C", "TIN"]:
        train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(size, padding=4), test_transform])
    else:
        train_transform = trn.Compose([trn.RandomResizedCrop(224), trn.RandomHorizontalFlip(), trn.ToTensor(), normalize])

    model = load_model(args, device, num_classes)

    if args.norm == "batch_norm":
        name = "no_bn.npy"
    elif args.norm == "instance_norm":
        if args.no_affine:
            name = "instance_norm_no_affine.npy"
        else:
            name = "instance_norm_affine.npy"
    elif args.norm == "group_norm":
        name = "group_norm_{}.npy".format(args.ngroups)
    elif args.norm == "identity":
        name = "identity.npy"
    else:
        raise NotImplementedError

    if args.dataset in ["CIFAR-10-C", "CIFAR-100-C", "TIN-C", "ImageNet-C"]:
        print("Starting default model")
        def_c_results = get_results_c(args, model, device, update_bn=False)
        np.save(os.path.join(save_dir, name), def_c_results)

        if args.norm == "batch_norm":
            print("Starting BN update without aug")
            bn_c_results = get_results_c(args, model, device, update_bn=True)
            np.save(os.path.join(save_dir, "def_bn.npy"), bn_c_results)

            print("Starting BN update with aug")
            tr_bn_c_results = get_results_c(args, model, device, update_bn=True, val_transform=train_transform)
            np.save(os.path.join(save_dir, "tr_bn.npy"), tr_bn_c_results)
    else:
        print("Starting default model")
        def_results = get_individual_results(args, model, device, update_bn=False)
        np.save(os.path.join(save_dir, name), def_results)

        if args.norm == "batch_norm":
            print("Starting BN update without aug")
            bn_results = get_individual_results(args, model, device, update_bn=True)
            np.save(os.path.join(save_dir, "def_bn.npy"), bn_results)

            print("Starting BN update with aug")
            tr_bn_results = get_individual_results(args, model, device, update_bn=True, val_transform=train_transform)
            np.save(os.path.join(save_dir, "tr_bn.npy"), tr_bn_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CIFAR-10-C")
    parser.add_argument("--norm", type=str, default="batch_norm")
    parser.add_argument("--ngroups", type=int, default=4, help="Number of groups if GroupNorm is used.")
    parser.add_argument("--no_affine", action="store_true", help="Whether to not use affine parameters for normalization.")
    parser.add_argument("--model", type=str, default="WideResNet")
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("--layers", default=40, type=int, help="total number of layers")
    parser.add_argument("--widen-factor", default=2, type=int, help="widen factor")
    parser.add_argument("--droprate", default=0.3, type=float, help="dropout probability")
    parser.add_argument("--ngpu", type=int, default=1, help="0 = CPU.")
    parser.add_argument("--load_dir", "-d", type=str, default="./snapshots", 
                        help="Directory containing model checkpoints", required=True)
    parser.add_argument("--root", type=str, default="./data", help="Directory containing datasets", required=True)
    args = parser.parse_args()
    main(args)

