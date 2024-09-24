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

from models.alexnet import AlexNet_cifar
from utils.model_trainer import TrainingManager
from models.layers.kDPFA import RConv2d as rADFAConv
import torchvision
from models.layers.GLobal_e import Ewrapper


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
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    return trainloader, testloader



def save_out_grad_hook(module, grad_input, grad_output):
    Ewrapper.update_E(grad_input)

def train(layer, lr, wd, ranks):
    
    device = 'cuda'
    dset_name = 'cifar10'
    total_classes = 10
    batch_size = 64
    vvs_depth=3
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
    
            
    epochs = 200
    vvs_layer = layer
    num_expirements = 3
    accuracies = dict()
    session_name = 'kt7tukuytk'
    max_lr = lr #5e-6
    total_steps = epochs
    #{'max_lr': max_lr, 'total_steps': total_steps, 'div_factor': 5, 'final_div_factor': 30} #epochs = 100
    for rank in ranks:
        accuracies[rank] = []
        for i in range(num_expirements):
            criterion = nn.CrossEntropyLoss()
            # criterion.register_full_backward_hook(save_out_grad_hook)
            model = AlexNet_cifar(1, kernel_size=9, bn = 32, num_classes=total_classes, device=device)
            model.vvs[0] = rADFAConv(32, 32, kernel_size=9, rank=rank, padding=9//2)
            model.vvs[3] = rADFAConv(32, 32, kernel_size=9, rank=rank, padding=9//2)
            # model.vvs[6] = rADFAConv(32, 32, kernel_size=9, rank=rank, padding=9//2)

            # model.vvs[1] = rADFAConv(32, 32, 9, rank=4, padding=9//2, out_dim=10)
            # model.vvs[2] = rADFAConv(32, 32, 9, rank=4, padding=9//2, out_dim=10)
            
            model.vvs[7].register_full_backward_hook(save_out_grad_hook)
            model.to(device)
            tm = TrainingManager(model,
                                trainloader,
                                testloader,
                                optim.Adam,
                                criterion,
                                epochs,
                                ExponentialLR,
                                rf"artifacts/{dset_name}/{session_name}/{vvs_layer}/r_{rank}/exp_{i}",
                                optimizer_params={'lr': max_lr, "weight_decay": wd},
                                scheduler_params={'gamma': 0.99},#, 'div_factor': 25, ,
                                device=device
                                )
            val_accuracy = tm.train_model()
            accuracies[rank].append(val_accuracy)
                
    import json
    with open(rf'artifacts/{dset_name}/{session_name}/{vvs_layer}/accuracies.json', 'w') as f:
        json.dump(accuracies, f)
        
        
        
if __name__ == "__main__":
    
    #for layer in ['vvs3']:
    # train('vvs1', 8e-6)
    train('vvs2', 1e-3, wd = 1e-5, ranks=[32])
    # train('vvs3', 1e-5, wd = 1e-6, ranks=[2, 8])
    # train('vvs3', 8e-6)
    # train('vvs2', 8e-6)