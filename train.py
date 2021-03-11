import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as trn
import torchvision.datasets as dset
from models.wrn import WideResNet, Identity

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


def train(model, train_loader, scheduler, optimizer, state, device):
    model.train()
    loss_avg = 0.0
    for i, (tr_data, tr_target) in enumerate(train_loader):
        tr_data, tr_target = tr_data.to(device), tr_target.long().squeeze().to(device)
        output = model(tr_data)

        # backward
        optimizer.zero_grad()
        loss = F.cross_entropy(output, tr_target)
        loss.backward()

        optimizer.step()
        scheduler.step()
        loss_avg += float(loss)
    loss_avg /= (i + 1)
    state["train_loss"] = loss_avg
    model.eval()

def test(model, test_loader, state, device):
    model.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            loss = F.cross_entropy(output, target)
            pred = output.data.max(1)[1]

            correct += pred.eq(target.data).sum().item()
            loss_avg += float(loss.data)
    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = 1.0 * correct / len(test_loader.dataset)

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    state = {k: v for k, v in args._get_kwargs()}
    print(state)
    exp_dir = os.path.join(args.snapshots_dir, "run{:02d}".format(args.run))
    # Make save directory
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if not os.path.isdir(exp_dir):
        raise Exception("%s is not a dir" % args.snapshots_dir)
    with open(os.path.join(exp_dir, "experiment_info.txt"), "a") as f:
        f.write("{}\n".format(args))

    device = "cpu" if args.ngpu == 0 else "cuda"

    test_transform = trn.Compose([trn.ToTensor()])
    if args.dataset == 'cifar10':
        size = 32
        train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(size, padding=4), trn.ToTensor()])
        train_data = dset.CIFAR10(args.data_dir, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(args.data_dir, train=False, transform=test_transform, download=True)
        num_classes = 10
        print("loaded CIFAR-10")
    elif args.dataset == 'cifar100':
        size = 32
        train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(size, padding=4), trn.ToTensor()])
        train_data = dset.CIFAR100(args.data_dir, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(args.data_dir, train=False, transform=test_transform, download=True)
        num_classes = 100
        print("loaded CIFAR-100")
    else:
        assert args.dataset == "tinyimagenet"
        size = 64
        train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(size, padding=4), trn.ToTensor()])
        train_data = dset.ImageFolder(os.path.join(args.data_dir, "tiny-imagenet-200", "train"), transform=train_transform)
        test_data = dset.ImageFolder(os.path.join(args.data_dir, "tiny-imagenet-200", "val"), transform=test_transform)
        num_classes = 200
        print("loaded TIN")

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.test_bs, shuffle=False,
        num_workers=args.prefetch, pin_memory=True)

    is_tinyimagenet = args.dataset == "tinyimagenet"
    norm = {"batch_norm": nn.BatchNorm2d, "group_norm": nn.GroupNorm, "instance_norm": nn.InstanceNorm2d,
            "layer_norm": nn.LayerNorm, "identity": Identity}[args.norm]
    model = WideResNet(args.layers, num_classes, widen_factor=args.widen_factor, dropRate=args.droprate,
                       tinyimagenet=is_tinyimagenet, norm=norm, ngroups=args.ngroups, affine=not args.no_affine)

    if args.ngpu > 1:
        model = nn.DataParallel(model, device_ids=list(range(args.ngpu)))
    model.to(device)
    if args.ngpu > 0:
        torch.cuda.manual_seed(args.seed)
        cudnn.benchmark = True  # fire on all cylinders

    optimizer = torch.optim.SGD(model.parameters(), state["learning_rate"], momentum=state["momentum"],
                                weight_decay=state["decay"], nesterov=True)

    print("Beginning Training\n")

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs * len(train_loader),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / args.learning_rate))

    for epoch in range(args.epochs):
        state["epoch"] = epoch
        begin_epoch = time.time()

        # train
        train(model, train_loader, scheduler, optimizer, state, device)

        # test
        test(model, test_loader, state, device)

        # Save model
        save_file = os.path.join(exp_dir, "checkpoint_epoch_" + str(epoch) + ".pt")
        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()},
            save_file)
        # Let us not waste space and delete the previous model unless epoch is a multiple of 25
        if not epoch % 25 == 0:
            prev_path = os.path.join(exp_dir, "checkpoint_epoch_" + str(epoch - 1) + ".pt")
            if os.path.exists(prev_path): os.remove(prev_path)

        # Show results
        with open(os.path.join(exp_dir, "training_results.csv"), "a") as f:
            f.write("%03d,%05d,%0.6f,%0.5f,%0.2f\n" % (
                (epoch + 1),
                time.time() - begin_epoch,
                state["train_loss"],
                state["test_loss"],
                100 - 100. * state["test_accuracy"],
            ))

        print("Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}".format(
            (epoch + 1),
            int(time.time() - begin_epoch),
            state["train_loss"],
            state["test_loss"],
            100 - 100. * state["test_accuracy"])
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.1, help="The initial learning rate.")
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size.")
    parser.add_argument("--test_bs", type=int, default=256)
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum.")
    parser.add_argument("--decay", "-d", type=float, default=0.0005, help="Weight decay (L2 penalty).")
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument("--layers", default=40, type=int, help="total number of layers")
    parser.add_argument("--widen-factor", default=2, type=int, help="widen factor")
    parser.add_argument("--droprate", default=0.3, type=float, help="dropout probability")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--prefetch", type=int, default=6, help="Pre-fetching threads.")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--norm", type=str, default="batch_norm")
    parser.add_argument("--ngroups", type=int, default=4, help="Number of groups if GroupNorm is used.")
    parser.add_argument("--no_affine", action="store_true", help="Whether to not use affine parameters for normalization.")
    parser.add_argument("--ngpu", type=int, default=1, help="0 = CPU.")
    parser.add_argument("--snapshots_dir", type=str, default="./snapshots", required=True)
    parser.add_argument("--data_dir", type=str, default="./data", required=True)
    parser.add_argument("--run", type=int, default=1)
    args = parser.parse_args()
    main(args)
