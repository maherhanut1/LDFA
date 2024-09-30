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

from models.FC import FC, FC_rAFA
from models.layers.GLobal_e import Ewrapper
from models.layers.rADFA_linear import Linear as rADFALinear
from models.layers.rADFA_linear import Linear as DFALInear
from utils.model_trainer import TrainingManager

import torchvision.datasets as datasets


LAYER_IDX = {
    'layer1': 0,
    'layer2': 3,
    'layer3': 6,
    'layer4': 10,
}

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
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

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
                                         shuffle=False, num_workers=2)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    
    
    return trainloader, testloader



def save_out_grad_hook(module, grad_input, grad_output):
    Ewrapper.update_E(grad_input)
        

def train_PFA_cifar10_exp_decay(session_name, layer, max_lr =1e-4, bn=512, ranks = [None], decay=1e-6):
    
    device = 'cuda'
    batch_size = 32
    total_classes = 10
    epochs= 160
    num_expirements = 5
    session_name = session_name
    dset_name = 'cifar10'
    trainloader, testloader = get_cifar10_loaders(batch_size=batch_size)
    all_accuracies = dict()
    # {'max_lr': max_lr, 'total_steps': epochs*len(trainloader), 'div_factor': 10, 'final_div_factor': 100}
    for i, rank in enumerate(ranks):
        max_lr = max_lr
        all_accuracies[rank] = []
        for i in range(num_expirements):
            criterion = nn.CrossEntropyLoss()
            model = FC_rAFA(input_dim=3*32*32, hidden_dim=bn, num_classes=total_classes)
            model.net[-1].register_full_backward_hook(save_out_grad_hook)
            model.net[LAYER_IDX['layer3']] = DFALInear(bn, bn, rank=rank, loss_module=criterion, update_P=True, update_Q=True, requires_gt=False)
            model.net[LAYER_IDX['layer2']] = DFALInear(bn, bn, rank=rank, loss_module=criterion, update_P=True, update_Q=True, requires_gt=False)
            model = model.to(device)
            tm = TrainingManager(model,
                            trainloader,
                            testloader,
                            optim.Adam,
                            criterion,
                            epochs,
                            ExponentialLR,
                            rf"/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset_name}/{session_name}/{layer}/r_{rank}/exp_{i}",
                            optimizer_params={'lr': max_lr, 'weight_decay': decay, 'amsgrad': True},
                            scheduler_params={'gamma': 0.975},  #0.97
                            device=device
                            )
            
            val_accuracy = tm.train_model()
            all_accuracies[rank].append(val_accuracy)
            
    import json
    
    
    with open(rf'/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset_name}/{session_name}/{layer}/accuracies.json', 'w') as f:
        json.dump(all_accuracies, f)
        


if __name__ == "__main__":
    # train_PFA_cifar10_exp_decay('DFA_V1', 'layer4', max_lr= 4e-4, bn=512, ranks=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], decay=5e-4)
    train_PFA_cifar10_exp_decay('DFA_from_layer_3_lr_6e4_wd_4e4', 'layer4', max_lr= 6e-4, bn=512, ranks=[32, 16, 10, 8, 6, 4, 2, 1], decay=4e-4)