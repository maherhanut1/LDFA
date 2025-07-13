import torch
from torch.optim.lr_scheduler import StepLR, OneCycleLR, ExponentialLR
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import numpy as np
import os

from models.ViT import ViT
from models.RAF_ViT import RafViT
from models.FC import FC_rAFA
from models.layers.rAFA_linear_att import Linear as rAFALinear
from utils.model_trainer import TrainingManager


from models.optim.AdeMamix import AdEMAMix
import torchvision.datasets as datasets

def get_mnist_loaders(batch_size=32):
    
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # MNIST is grayscale, single channel
])

    # Downloading/loading the MNIST training dataset
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Downloading/loading the MNIST test dataset
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return trainloader, testloader

def get_cifar10_loaders(batch_size=32):
    transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomRotation(10),  # Randomly rotate images by up to 10 degrees
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return trainloader, testloader

def get_cifar100_loaders(batch_size=32, class_limit=1000):
    transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomRotation(5),  # Randomly rotate images by up to 10 degrees
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    test_idx = [i for i in range(len(testset.targets)) if testset.targets[i] < class_limit]
    train_idx = [i for i in range(len(trainset.targets)) if trainset.targets[i] < class_limit]
    testset = torch.utils.data.Subset(testset, test_idx)
    trainset = torch.utils.data.Subset(trainset, train_idx)
    testloader = DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=1)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    return trainloader, testloader
    

def train_PFA_cifar10_gen(class_type, session_name, ranks=[10], max_lr =8e-6, bn=512, decay=1e-6):
    
    device = 'cuda'
    batch_size = 128
    total_classes = 10
    epochs= 150
    num_expirements = 5
    session_name = session_name
    dset_name = 'cifar10'
    trainloader, testloader = get_cifar10_loaders(batch_size=batch_size)
    all_accuracies = dict()
    
    for rank in ranks:
        all_accuracies[rank] = []
        for i in range(num_expirements):
            model = class_type(3, 10, 32, 8, hidden=384, mlp_hidden=384*2, dropout=0.1)
            # model = FC_rAFA()
                
            model.to(device)
            max_lr = max_lr
            tm = TrainingManager(model,
                            trainloader,
                            testloader,
                            optim.Adam,
                            nn.CrossEntropyLoss(),
                            epochs,
                            ExponentialLR,
                            rf"/home/maherhanut/Documents/AFA/artifacts/{dset_name}/{session_name}/RAF/r_{rank}/exp_{i}",
                            optimizer_params={'lr': max_lr, 'weight_decay': decay},
                            scheduler_params={'gamma': 0.98},
                            device=device
                            )
            
            val_accuracy = tm.train_model()
            all_accuracies[rank].append(val_accuracy)
            
    import json
    
    
    with open(rf'/home/maherhanut/Documents/AFA/artifacts/{dset_name}/{session_name}/RAF/accuracies.json', 'w') as f:
        json.dump(all_accuracies, f)

if __name__ == "__main__":
    
    
    #set 1
    lrs_set2 = [3e-4]
    # lrs_set1 = [5e-5, 1e-5]
    decays = [1e-4]
    
    
    train_PFA_cifar10_gen(RafViT, f'RAFVIT_rank_16', max_lr=8e-4, bn=512, decay=1e-4, ranks=[16])