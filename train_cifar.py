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

from models.alexnet import AlexNet_cifar, AlexNet_AFA
from utils.model_trainer import TrainingManager
from models.layers.rAFA_conv import Conv2d as rAFAConv
import torchvision


def get_cifar_10_loader(batch_size=32):
    
    transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert Images to Grayscale
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomRotation(10),  # Randomly rotate images by up to 10 degrees
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize images
])
    
    transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert Images to Grayscale
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize images
])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    return trainloader, testloader

def train(session_name, layer, lr, wd, ranks, gamma):
    
    device = 'cuda'
    dset_name = 'cifar10'
    total_classes = 10
    batch_size = 64
    trainloader, testloader = get_cifar_10_loader(batch_size=batch_size)
    vvs_idx_dict = {
        'vvs1': 0,
        'vvs2': 3,
        'vvs3': 6,
    }
    
            
    epochs = 120
    vvs_layer = layer
    num_expirements = 1
    accuracies = dict()
    max_lr = lr #5e-6
    #{'max_lr': max_lr, 'total_steps': total_steps, 'div_factor': 5, 'final_div_factor': 30} #epochs = 100
    for rank in ranks:
        accuracies[rank] = []
        for i in range(num_expirements):
            model = AlexNet_AFA(1, kernel_size=9, bn = 32, num_classes=total_classes, device=device, update_p=True, update_q=True)
            model.vvs[0] = rAFAConv(32, 32, 9, rank=rank, padding=9//2, update_p=True, update_q=True)
            model.to(device)

            tm = TrainingManager(model,
                                trainloader,
                                testloader,
                                optim.RMSprop,
                                nn.CrossEntropyLoss(),
                                epochs,
                                ExponentialLR,
                                rf"artifacts/{dset_name}/{session_name}/{vvs_layer}/r_{rank}/exp_{i}",
                                optimizer_params={'lr': max_lr, "weight_decay": wd},
                                scheduler_params={'gamma': gamma},
                                device=device
                                )
            val_accuracy = tm.train_model()
            accuracies[rank].append(val_accuracy)
                
    import json
    with open(rf'artifacts/{dset_name}/{session_name}/{vvs_layer}/accuracies.json', 'w') as f:
        json.dump(accuracies, f)
        
        

def train_const_all(session_name, layer, lr, wd, ranks, with_p = True):
    
    device = 'cuda'
    dset_name = 'cifar10'
    total_classes = 10
    batch_size = 64
    trainloader, testloader = get_cifar_10_loader(batch_size=batch_size)
    # vvs_idx_dict = {
    #     'vvs1': 0,
    #     'vvs2': 2,
    #     'vvs3': 4,
    # }
    vvs_idx_dict = {
        'vvs1': 0,
        'vvs2': 3,
        'vvs3': 6,
    }
    

    epochs = 250
    vvs_layer = layer
    num_expirements = 5
    accuracies = dict()
    max_lr = lr #5e-6
    #{'max_lr': max_lr, 'total_steps': total_steps, 'div_factor': 5, 'final_div_factor': 30} #epochs = 100
    for rank in ranks:
        accuracies[rank] = []
        for i in range(num_expirements):
            model = AlexNet_AFA(1, kernel_size=9, bn = 32, num_classes=total_classes, device=device, constrain = rank, update_p=with_p, update_q=True)
            # model.vvs[vvs_idx_dict[layer]] = rAFAConv(32, 32, 9, rank=rank, padding=9//2)
            model.to(device)

            tm = TrainingManager(model,
                                trainloader,
                                testloader,
                                optim.Adam,
                                nn.CrossEntropyLoss(),
                                epochs,
                                ExponentialLR,
                                rf"artifacts/{dset_name}/{session_name}/{vvs_layer}/r_{rank}/exp_{i}",
                                optimizer_params={'lr': max_lr, "weight_decay": wd},
                                scheduler_params={'gamma': 0.98},
                                device=device
                                )
            val_accuracy = tm.train_model()
            accuracies[rank].append(val_accuracy)
                
    import json
    with open(rf'artifacts/{dset_name}/{session_name}/{vvs_layer}/accuracies.json', 'w') as f:
        json.dump(accuracies, f)
        
def train_BP(session_name, lr, wd, gamma):
    
    device = 'cuda'
    dset_name = 'cifar10'
    total_classes = 10
    batch_size = 64
    trainloader, testloader = get_cifar_10_loader(batch_size=batch_size)

    epochs = 120
    num_expirements = 2
    accuracies = dict()
    max_lr = lr #5e-6
    #{'max_lr': max_lr, 'total_steps': total_steps, 'div_factor': 5, 'final_div_factor': 30} #epochs = 100
    accuracies['BP'] = []
    for i in range(num_expirements):
        model = AlexNet_cifar(1, bn=4, kernel_size=9, num_classes=total_classes, device=device)
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
                            scheduler_params={'gamma': gamma},
                            device=device
                            )
        val_accuracy = tm.train_model()
        accuracies['BP'].append(val_accuracy)
                
    import json
    with open(rf'artifacts/{dset_name}/{session_name}/BP/accuracies.json', 'w') as f:
        json.dump(accuracies, f)
        
        
        
        
if __name__ == "__main__":
    
    # print('constraint all')
    
    # train_const_all('constraint_all_v3', 'constrain_all_update_QP', 5e-4, wd = 5e-5, ranks=[32, 16, 8, 4, 2, 1], with_p=True)
    
    
    # train('Retina_model_constrain_retina_only_lr_1e4_wd_1e5_gamma_98', 'vvs1', lr=1e-4, wd=1e-5, ranks=[4, 32], gamma=0.98)
    train_BP('Retina_model_constrain_retina_only_lr_1e4_wd_1e5_gamma_98', lr=1e-4, wd=1e-5, gamma=0.98)