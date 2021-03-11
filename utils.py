import os
import time
import math
import numbers
import random
import numpy as np
import torch
import torchvision
import torchvision.datasets as dset
import torch.utils.data as data
import torchvision.transforms as trn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
from models.wrn import WideResNet, Identity

import numpy.random as rnd
import seaborn as sns
import numpy.linalg as la
import scipy.linalg as sla
from scipy import stats

test_corruptions = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression"]

class CorruptionDataset(Dataset):
    def __init__(self, root="", transform=None, val=True, cifar10=True, severity=-1, distortion=None):
        """
        if severity == -1, then uses all severities. 
        """
        dir = "CIFAR-10-C" if cifar10 else "CIFAR-100-C"
        self.root = os.path.join(root, dir)
        labels = np.load(os.path.join(self.root, "labels.npy"))
        
        k = 10000
        if severity == -1:
            start, end = 0, 5*k
        else:
            start, end = k*(severity-1), k*severity
        
        dir = os.path.join(self.root, "extra") if val else self.root
        
        if distortion is None:
            self.data = np.concatenate([np.load(os.path.join(dir, "{}.npy".format(corruption)))[start:end] for corruption in test_corruptions])
            self.labels = np.concatenate([labels for _ in range(len(test_corruptions))])
        else:
            self.data = np.load(os.path.join(dir, "{}.npy".format(distortion)))[start:end]
            self.labels = labels

        self.transform = transform
        if self.transform is None:
            self.transform = trn.Compose([trn.ToTensor()])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        img = self.data[item].astype(np.uint8)
        return self.transform(img), self.labels[item]
    
def load_model(args, device, num_classes):
    if args.dataset in ["ImageNet", "ImageNet-C", "ImageNet-v2", "StylizedImageNet"]:
        if args.model == "WideResNet":
            model = torchvision.models.wide_resnet50_2(pretrained=True).to(device)
        elif args.model == "ResNet-50":
            model = torchvision.models.resnet50(pretrained=True).to(device)
        else:
            raise NotImplementedError

        if args.ngpu > 0:
            torch.cuda.manual_seed(1)
            cudnn.benchmark = True
            model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
        return model

    is_tin = args.dataset in ["tinyimagenet", "TIN-C", "TIN"]
    assert args.model == "WideResNet"

    norm = {"batch_norm": nn.BatchNorm2d, "group_norm": nn.GroupNorm, "instance_norm": nn.InstanceNorm2d,
            "layer_norm": nn.LayerNorm, "identity": Identity}[args.norm]
    model = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate, tinyimagenet=is_tin,
                       norm=norm, ngroups=args.ngroups, affine=not args.no_affine)

    if args.ngpu > 0:
        torch.cuda.manual_seed(1)
        cudnn.benchmark = True
        model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
    model.to(device)

    if args.dataset in ["TIN-C", "TIN"]:
        dataset = "TIN"
    elif args.dataset in ["CIFAR-10-C", "CIFAR-10"]:
        dataset = "CIFAR-10"
    elif args.dataset in ["CIFAR-100-C", "CIFAR-100"]:
        dataset = "CIFAR-100"
    else:
        raise NotImplementedError

    if args.norm == "batch_norm":
        norm_dir = "batch_norm"
    elif args.norm == "group_norm":
        norm_dir = "group_norm_{}".format(args.ngroups)
    elif args.norm == "instance_norm":
        if args.no_affine:
            norm_dir = "instance_norm_no_affine"
        else:
            norm_dir = "instance_norm_affine"
    elif args.norm == "identity":
        norm_dir = "identity"
    else:
        raise NotImplementedError

    load_file = os.path.join(args.load_dir, dataset, norm_dir, "checkpoint_epoch_99.pt")
    checkpoint = torch.load(load_file)
    state_dict = checkpoint["model_state_dict"]
    try:
        model.module.load_state_dict(state_dict)
    except:
        model.load_state_dict(state_dict)
    return model

def get_labels(loader):
    labels = []
    for i, (_, target) in enumerate(loader):
        labels.append(target.cpu().numpy())
    labels = np.concatenate(labels)
    return labels

def get_model_bn_stats(model):
    means = []
    vars = []

    def get_bn_stats(m):
        if type(m) is nn.BatchNorm2d:
            means.append(m.running_mean.cpu().numpy())
            vars.append(m.running_var.cpu().numpy())

    model.apply(get_bn_stats)
    return means, vars

def get_bn_stats(std_model, perturbed_model, concatentate=False):
    std_running_means = []
    perturbed_running_means = []
    std_running_vars = []
    perturbed_running_vars = []
    names = []
    for p1, p2 in zip(std_model.state_dict(), perturbed_model.state_dict()):
        assert p1 == p2
        if "mean" in p1:
            names.append(p1)
            mean1 = std_model.state_dict()[p1].cpu().numpy().flatten()
            std_running_means.append(mean1)
            mean2 = perturbed_model.state_dict()[p2].cpu().numpy().flatten()
            perturbed_running_means.append(mean2)
        elif "var" in p1:
            names.append(p1)
            var1 = std_model.state_dict()[p1].cpu().numpy().flatten()
            std_running_vars.append(var1)
            var2 = perturbed_model.state_dict()[p2].cpu().numpy().flatten()
            perturbed_running_vars.append(var2)
        else:
            diff = (std_model.state_dict()[p1] - perturbed_model.state_dict()[p2]).sum().item()
            if "num_batches_tracked" not in p1:
                assert diff == 0
    # turn into a numpy array
    if concatentate:
        std_running_means = np.concatenate(std_running_means)
        perturbed_running_means = np.concatenate(perturbed_running_means)
        std_running_vars = np.concatenate(std_running_vars)
        perturbed_running_vars = np.concatenate(perturbed_running_vars)
    return std_running_means, perturbed_running_means, std_running_vars, perturbed_running_vars, names

def plot_bn_stats(std_means, perturbed_means, std_vars, perturbed_vars, save_dirs, concatenated=True):
    if not concatenated:
        std_means = np.concatenate(std_means)
        perturbed_means = np.concatenate(perturbed_means)
        std_vars = np.concatenate(std_vars)
        perturbed_vars = np.concatenate(perturbed_vars)

    fig, axes = plt.subplots(2, figsize=(10, 20))
    
    begin, end = -1, 1.2
    xs = np.arange(begin, end, step=0.1)
    ys = np.arange(begin, end, step=0.1)
    axes[0].plot(xs, ys, "-")
    axes[0].scatter(std_means, perturbed_means)
    axes[0].set_title("Running means")
    axes[0].set_xlabel("Original")
    axes[0].set_ylabel("Perturbed")

    begin, end = 0, 1
    xs = np.arange(begin, end, step=0.1)
    ys = np.arange(begin, end, step=0.1)
    axes[1].plot(xs, ys, "-")
    axes[1].scatter(std_vars, perturbed_vars)
    axes[1].set_title("Running vars")
    axes[1].set_xlabel("Original")
    axes[1].set_ylabel("Perturbed")
    
    for dir in save_dirs:
        file = os.path.join(dir, "bn_stats")
        plt.savefig(file)
    plt.clf()
        

def calib_err(confidence, correct, p='2', beta=100):
    # beta is target bin size
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0
    total_examples = len(confidence)
    for i in range(len(bins) - 1):
        bin_confidence = confidence[bins[i][0]:bins[i][1]]
        bin_correct = correct[bins[i][0]:bins[i][1]]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))

            if p == '2':
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == '1':
                cerr += num_examples_in_bin / total_examples * difference
            elif p == 'infty' or p == 'infinity' or p == 'max':
                cerr = np.maximum(cerr, difference)
            else:
                assert False, "p must be '1', '2', or 'infty'"

    if p == '2':
        cerr = np.sqrt(cerr)

    return cerr


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

normalize = trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
imagenet_transform = trn.Compose([
    trn.Resize(256),
    trn.CenterCrop(224),
    trn.ToTensor(),
    normalize,
])

def load_corrupted_dataset(dataset, root, tr, sev, transform=None):
    assert dataset in ["CIFAR-10-C", "CIFAR-100-C", "TIN-C", "ImageNet-C"]
    if dataset == "CIFAR-10-C":
        return CorruptionDataset(root=root, distortion=tr, val=False, cifar10=True, severity=sev, transform=transform)
    elif dataset == "CIFAR-100-C":
        return CorruptionDataset(root=root, distortion=tr, val=False, cifar10=False, severity=sev, transform=transform)
    elif dataset == "TIN-C":
        if transform is None:
            return dset.ImageFolder(os.path.join(root, "Tiny-ImageNet-C", tr, "{}".format(sev)),
                                    transform=trn.Compose([trn.ToTensor()]))
        else:
            return dset.ImageFolder(os.path.join(root, "Tiny-ImageNet-C", tr, "{}".format(sev)), transform=transform)
    elif dataset == "ImageNet-C":
        if transform is None:
            return dset.ImageFolder(os.path.join(root, "ImageNet-C", tr, "{}".format(sev)), transform=imagenet_transform)
        else:
            return dset.ImageFolder(os.path.join(root, "ImageNet-C", tr, "{}".format(sev)), transform=transform)
    else:
        raise NotImplementedError

def get_dataset(args, transform=None):
    if args.dataset in ["CIFAR-10", "CIFAR-10-C"]:
        if transform is None:
            transform = trn.Compose([trn.ToTensor()])
        test_dataset = dset.CIFAR10(args.root, train=False, transform=transform, download=True)
        num_classes = 10
        size = 32
    elif args.dataset in ["CIFAR-100", "CIFAR-100-C"]:
        if transform is None:
            transform = trn.Compose([trn.ToTensor()])
        test_dataset = dset.CIFAR100(args.root, train=False, transform=transform, download=True)
        num_classes = 100
        size = 32
    elif args.dataset in ["TIN", "TIN-C"]:
        if transform is None:
            transform = trn.Compose([trn.ToTensor()])
        test_dataset = dset.ImageFolder(os.path.join(args.root, "tiny-imagenet-200", "val"), transform=transform)
        num_classes = 200
        size = 64
    else:
        if transform is None:
            normalize = trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transform = trn.Compose([
                trn.Resize(256),
                trn.CenterCrop(224),
                trn.ToTensor(),
                normalize,
            ])
        test_dataset = dset.ImageFolder(os.path.join(args.root, "ILSVRC2012", "val"), transform=transform)
        num_classes = 1000
        size = 224
    return test_dataset, transform, num_classes, size

def select_classes_dataset(dataset, keep_classes, num_classes=10):
    labels = get_labels(torch.utils.data.DataLoader(dataset, batch_size=1028)).squeeze()
    idxs = torch.zeros(len(labels)).bool()
    for k in range(num_classes):
        if k in keep_classes:
            idxs[labels == k] = True
    indices = list(np.argwhere(idxs.flatten()).flatten())
    return torch.utils.data.Subset(dataset, indices=indices)

def get_num_classes(dataset):
    if dataset in ["cifar10", "CIFAR-10", "CIFAR-10-C"]:
        return 10
    if dataset in ["cifar100", "CIFAR-100", "CIFAR-100-C"]:
        return 100
    if dataset in ["tinyimagenet", "TIN", "TIN-C"]:
        return 200
    else:
        return 1000
