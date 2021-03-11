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
import sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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
from utils import calib_err, load_model, get_dataset, select_classes_dataset, get_num_classes


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

def get_black_border_results(args, orig_model, device, val_dataset, bn_val_dataset, update_bn=False,
                             idxs=None, keep_classes=None):
    num_classes = get_num_classes(args.dataset)
    if keep_classes is not None:
        val_dataset = select_classes_dataset(val_dataset, keep_classes, num_classes)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True)

    if keep_classes is not None:
        bn_val_dataset = select_classes_dataset(bn_val_dataset, keep_classes, num_classes)
    bn_val_loader = torch.utils.data.DataLoader(bn_val_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True)

    if args.norm == "batch_norm" and update_bn:
        model = update_bn_stats(orig_model, bn_val_loader, device, idxs)
    else:
        model = orig_model

    acc, conf, ce = get_acc_conf_ce(model, device, val_loader)
    print("acc {:.3f} conf {:.3f} ce {:.3f}".format(acc, conf, ce))

    results = {"accs": acc, "confs": conf, "cerrs": ce}
    return results

normalize = trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
imagenet_transform = trn.Compose([
    trn.Resize(256),
    trn.CenterCrop(224),
    trn.ToTensor(),
    normalize,
])

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
    save_dir = os.path.join("final_results", "black_border_evaluation", args.dataset)
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    test_dataset, test_transform, num_classes, size = get_dataset(args)

    severity = size // 4

    black_border_transform = trn.Compose([trn.CenterCrop(size - 2 * severity), trn.Pad(severity), trn.ToTensor()])

    if args.dataset in ["CIFAR-10", "TIN-C", "TIN"]:
        black_border_tr_transform = trn.Compose(
            [trn.CenterCrop(size - 2 * severity), trn.Pad(severity),trn.RandomHorizontalFlip(),
             trn.RandomCrop(size, padding=4), trn.ToTensor()])
    else:
        in_size = 256
        black_border_tr_transform = trn.Compose(
            [trn.CenterCrop(in_size - 2 * severity), trn.Pad(severity), trn.RandomHorizontalFlip(),
             trn.RandomCrop(in_size, padding=4), trn.RandomResizedCrop(224), trn.RandomHorizontalFlip(), trn.ToTensor(),
             normalize])

    val_dataset, _, _, _ = get_dataset(args, transform=black_border_transform)
    tr_val_dataset, _, _, _ = get_dataset(args, transform=black_border_tr_transform)

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

    print("Starting default model")
    def_results = get_black_border_results(args, model, device, val_dataset, val_dataset, update_bn=False)
    np.save(os.path.join(save_dir, name), def_results)

    if args.norm == "batch_norm":
        print("Starting BN update without aug")
        bn_results = get_black_border_results(args, model, device, val_dataset, val_dataset, update_bn=True)
        np.save(os.path.join(save_dir, "def_bn.npy"), bn_results)

        print("Starting BN update with aug")
        tr_bn_results = get_black_border_results(args, model, device, val_dataset, tr_val_dataset, update_bn=True)
        np.save(os.path.join(save_dir, "tr_bn.npy"), tr_bn_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CIFAR-10")
    parser.add_argument("--norm", type=str, default="batch_norm")
    parser.add_argument("--ngroups", type=int, default=4, help="Number of groups if GroupNorm is used.")
    parser.add_argument("--no_affine", action="store_true", help="Whether to not use affine parameters for normalization.")
    parser.add_argument("--model", type=str, default="WideResNet")
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("--layers", default=40, type=int, help="total number of layers")
    parser.add_argument("--widen-factor", default=2, type=int, help="widen factor")
    parser.add_argument("--droprate", default=0.3, type=float, help="dropout probability")
    parser.add_argument("--ngpu", type=int, default=1, help="0 = CPU.")
    parser.add_argument("--use_norm_zero", action="store_true")
    parser.add_argument("--load_dir", "-d", type=str, default="./snapshots", required=True)
    parser.add_argument("--root", type=str, default="./data", required=True)
    args = parser.parse_args()
    main(args)

