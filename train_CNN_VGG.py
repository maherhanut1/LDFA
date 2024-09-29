import torch
from torch.optim.lr_scheduler import StepLR, OneCycleLR, ExponentialLR, CosineAnnealingLR
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import numpy as np
import os

from models.alexnet import AlexNet_cifar, AlexNet_AFA
from models.VGG16 import CIFAR10CNN, CIFAR10CNNBP
from utils.model_trainer import TrainingManager
from models.layers.rAFA_conv import Conv2d as rAFAConv
import torchvision

from models.layers.GLobal_e import Ewrapper

mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]


class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes  # Number of regions to cut out
        self.length = length    # Length of the square region

    def __call__(self, img):
        h, w = img.size(1), img.size(2)

        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def get_cifar_10_loader(batch_size=32):
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        Cutout(n_holes=1, length=16),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    return trainloader, testloader


def train(session_name, lr, wd, ratio, update_backward):
    
    device = 'cuda'
    dset_name = 'cifar10'
    total_classes = 10
    batch_size = 64
    trainloader, testloader = get_cifar_10_loader(batch_size=batch_size)
    

    epochs = 400
    num_expirements = 10
    accuracies = dict()
    max_lr = lr #5e-6
    #{'max_lr': max_lr, 'total_steps': total_steps, 'div_factor': 5, 'final_div_factor': 30} #epochs = 100
    accuracies['constraint'] = []
    for i in range(num_expirements):
        
        if ratio == -1:
            model = CIFAR10CNNBP()
        else:
            model = CIFAR10CNN(rank_ratio = ratio, update_backward=update_backward)
        # model.vvs[vvs_idx_dict[layer]] = rAFAConv(32, 32, 9, rank=rank, padding=9//2)
        model.to(device)

        tm = TrainingManager(model,
                            trainloader,
                            testloader,
                            optim.Adam,
                            nn.CrossEntropyLoss(),
                            epochs,
                            ExponentialLR,
                            rf"artifacts/{dset_name}/{session_name}/BP/exp_{i}",
                            optimizer_params={'lr': max_lr, "weight_decay": wd},
                            scheduler_params={'gamma': 0.98},
                            device=device
                            )
        val_accuracy = tm.train_model()
        accuracies['constraint'].append(val_accuracy)
                
    import json
    with open(rf'artifacts/{dset_name}/{session_name}/accuracies.json', 'w') as f:
        json.dump(accuracies, f)

        
        
        
if __name__ == "__main__":
    
    # print('constraint all')
    # train_const_all('constraint_all_v2', 'constrain_all_update_QP', 3e-4, wd = 5e-5, ranks=[16, 8, 4, 2, 1], with_p=True)
    
    
    # train_BP('vgg16rAFA_x2_lr_5e4_wd_5e5_gamma_98', 5e-4, wd = 5e-5, ratio = 2)
    # train_BP('vgg16rAFA_x4_lr_5e4_wd_5e5_gamma_98', 5e-4, wd = 5e-5, ratio = 4)
    # train_BP('vgg16rAFA_x8_lr_5e4_wd_5e5_gamma_98', 5e-4, wd = 5e-5, ratio = 8)
    
    
    
    train('vgg16r_standard_FA_x1_lr_5e4_wd_5e5_gamma_98', 5e-4, wd = 5e-5, ratio = 1, update_backward=False)
    train('vgg16r_standard_FA_x2_lr_5e4_wd_5e5_gamma_98', 5e-4, wd = 5e-5, ratio = 2, update_backward=False)
    # train('vgg16r_standard_FA_x4_lr_5e4_wd_5e5_gamma_98', 5e-4, wd = 5e-5, ratio = 4, update_backward=False)
    # train('vgg16r_standard_FA_x8_lr_5e4_wd_5e5_gamma_98', 5e-4, wd = 5e-5, ratio = 8, update_backward=False)
    
