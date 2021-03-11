# Limitations of Post-Hoc Feature Alignment for Robustness
---
This is the code for the CVPR 2021 paper [Limitations of Post-Hoc Feature Alignment for Robustness](https://arxiv.org/abs/2103.05898) by Collin Burns and Jacob Steinhardt.


## Abstract
---
Feature alignment is an approach to improving robustness to distribution shift that matches the distribution of feature activations between the training distribution and test distribution. A particularly simple but effective approach to feature alignment involves aligning the batch normalization statistics between the two distributions in a trained neural network. This technique has received renewed interest lately because of its impressive performance on robustness benchmarks. However, when and why this method works is not well understood. We investigate the approach in more detail and identify several limitations. We show that it only significantly helps with a narrow set of distribution shifts and we identify several settings in which it even degrades performance. We also explain why these limitations arise by pinpointing why this approach can be so effective in the first place. Our findings call into question the utility of this approach and Unsupervised Domain Adaptation more broadly for improving robustness in practice.


## Dependencies
---
The code has the following requirements:
- Python 3+
- PyTorch
- Torchvision

It also requires several datasets:
The datasets used are:
- [CIFAR-10-C, TinyImageNet-C, and ImageNet-C](https://github.com/hendrycks/robustness/tree/master/ImageNet-C)
- [ImageNet-v2](https://github.com/modestyachts/ImageNetV2)
- [Stylized ImageNet](https://github.com/rgeirhos/Stylized-ImageNet) (This has additional dependencies)

Download the relevant datasets to ./data, or change the --root flag when running each script. 

## Overview of Scripts
---
- train.py trains a CIFAR-10, CIFAR-100, or TinyImageNet model from scratch using a WideResNet model (provided in models/wrn.py). Pre-trained models for CIFAR-10 and TinyImageNet are provided in the snapshots folder.
- evaluate_all.py tests a model on a shifted dataset, and saves the accuracies, calibration errors, and (optionally) the Batch Norm statistics after applying AdaBN.
- label_shift_experiment.py tests AdaBN when applied to only a subset of classes. It also tests how this varies when one only updates the statistics for a subset of layers.
- evaluate_black_border.py tests AdaBN on the "Black border" transformation, for which AdaBN degrades performance.

When testing on an ImageNet dataset, make sure to use the flag: --model ResNet-50

## Citation
If you find this useful in your research, please consider citing:

    @article{burns2021limitations,
          title={Limitations of Post-Hoc Feature Alignment for Robustness}, 
          author={Collin Burns and Jacob Steinhardt},
          journal={Conference on Computer Vision and Pattern Recognition (CVPR)},
          year={2021}
    }
