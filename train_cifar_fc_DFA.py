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
    epochs= 80
    num_expirements = 10
    session_name = session_name
    dset_name = 'cifar10'
    trainloader, testloader = get_cifar10_loaders(batch_size=batch_size)
    all_accuracies = dict()
    # {'max_lr': max_lr, 'total_steps': epochs*len(trainloader), 'div_factor': 10, 'final_div_factor': 100}
    for i, rank in enumerate(ranks):
        max_lr = max_lrs[i]
        all_accuracies[rank] = []
        for i in range(num_expirements):
            criterion = nn.CrossEntropyLoss()
            criterion.register_full_backward_hook(save_out_grad_hook)
            model = FC(input_dim=3*32*32, hidden_dim=bn, num_classes=total_classes)
            model.net[LAYER_IDX['layer4']] = rADFALinear(bn,total_classes, rank=rank, loss_module=criterion, update_Q=True, update_P=True, requires_gt=True)
            model.net[LAYER_IDX['layer3']] = rADFALinear(bn, bn, rank=rank, loss_module=criterion, update_P=True, update_Q=True, requires_gt=True)
            model.net[LAYER_IDX['layer2']] = rADFALinear(bn, bn, rank=rank, loss_module=criterion, update_P=True, update_Q=True, requires_gt=True)
            model = model.to(device)
            tm = TrainingManager(model,
                            trainloader,
                            testloader,
                            optim.Adam,
                            criterion,
                            epochs,
                            ExponentialLR,
                            rf"/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset_name}/{session_name}/{layer}/r_{rank}/exp_{i}",
                            optimizer_params={'lr': max_lr, 'weight_decay': decay},
                            scheduler_params={'gamma': 0.96},  #0.97
                            device=device
                            )
            
            val_accuracy = tm.train_model()
            all_accuracies[rank].append(val_accuracy)
            
    import json
    
    
    with open(rf'/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset_name}/{session_name}/{layer}/accuracies.json', 'w') as f:
        json.dump(all_accuracies, f)
        


if __name__ == "__main__":
    # train_PFA_cifar10_constraint_all('rAFA_512_x4_batch_norm_constrain_all_20_30_50', max_lr=2e-5, bn=512, decay=1e-6)
    # train_PFA_cifar10_no_constraint('rAFA_512_x4_no_constraint', 8e-6, bn=512, decay=1e-6)
    
    # train_PFA_cifar10_BP('BP_512', max_lr=8e-6, bn=512, decay=1e-6)
    # train_PFA_cifar10_BP('BP_128', max_lr=8e-6, bn=128, decay=1e-6)
    # train_PFA_cifar10_no_constraint('rAFA_128_x4_no_constraint', 8e-6, bn=128, decay=1e-6)
    
    
    # cifar_subset_ranks = [1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 128, 256][::-1]
    

    
    #cifar100_ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512][::-1]
    # cifar_10_128_ranks = [1, 2, 4, 8, 16, 32, 64, 128][::-1]
  #  cifar10_ranks = [1, 2, 4, 8, 16, 32, 64]
    
   
    #train cifar on 20 classes
    # train_PFA_cifar100_subsets('rAFA_one_layer_512_x3_subsets', 'layer3', max_lr=8e-6, bn=512, ranks=cifar_subset_ranks, decay=1e-6, epochs=80, pct_start=0.2, class_limit=100)
    


#for cifar10 512 lr used is 8e-6
#for cifar10 128 lr used is 1e-5, for layer4 2e-5 with grad normalization


#forcifar100 512x5 lr used for layer3: 1.3e-5, layer4:1.5e-5 , layer2:1e-5

#cifar50 3 layers LR FOR LAYER2: 5e-6, layer3: 8e-5

##################### Final Commands ############################################

########################################### train on cifar10 512, 128 neurons x 4 layers x ranks with batchnorm ###########################################
    


    # train_PFA_cifar10_BP('2e4_expdecay_96_BP', max_lr=2e-4, bn=512, decay=1e-6)
    ranks = [64, 32, 20, 16, 8, 4, 2, 1]
    max_lrs = np.linspace(1e-4, 3e-4, len(ranks))[::-1]
    train_PFA_cifar10_exp_decay('test_DFA', 'layer4', max_lr= max_lrs, bn=512, ranks=ranks, decay=1e-6)
    # train_PFA_cifar10_exp_decay('update_pq_2e4_expdecay_96_nogt', 'layer3', max_lr= 2e-4, bn=512, ranks=[64, 32, 20, 16, 8, 4, 2, 1][::-1], decay=1e-6)
    # train_PFA_cifar10_exp_decay('update_pq_2e4_expdecay_96_nogt', 'layer2', max_lr= 2e-4, bn=512, ranks=[64, 32, 20, 16, 8, 4, 2, 1][::-1], decay=1e-6)
    
    # train_PFA_cifar10_constraint_all('update_pq_8e5_expdecay_96_nogt_constraint_norm_32_32_32', max_lr=1e-3, bn=512, decay=1e-6)
    # train_PFA_cifar10('rAFA_one_layer_128_x4_batch_norm', 'layer2', max_lr=8e-6, bn=128, ranks=[64, 32, 20, 16, 8, 4, 2, 1], decay=1e-6)
######################################################################################################################################################

########################################### train on cifar10 512 neurons x 4 layers x constrain all layers ranks, BP, and no constraints ########################################### 
    # train_PFA_cifar10_no_constraint('rAFA_one_layer_512_x4_no_constraint_batch_norm', max_lr=8e-6, bn=512, decay=1e-6)
    # train_PFA_cifar10_constraint_all('rAFA_one_layer_512_x4_constraint_all_batch_norm_32_64_128', max_lr=7e-6, bn=512, decay=1e-6)
    # train_PFA_cifar10_constraint_all('rAFA_one_layer_512_x4_constraint_all_batch_norm_64_64_128', max_lr=7e-6, bn=512, decay=1e-6)
    # train_PFA_cifar10_constraint_all('rAFA_one_layer_512_x4_constraint_all_batch_norm_16_96_128', max_lr=7e-6, bn=512, decay=1e-6)
    
    
    # train_PFA_cifar10_constraint_all('rAFA_one_layer_512_x4_constraint_all_batch_norm_64_128_128', max_lr=7e-6, bn=512, decay=1e-6)
    # train_PFA_cifar10_constraint_all('rAFA_one_layer_512_x4_constraint_all_batch_norm_32_32_128', max_lr=7e-6, bn=512, decay=1e-6)
    # train_PFA_cifar10_constraint_all('rAFA_one_layer_512_x4_constraint_all_batch_norm_128_128_128', max_lr=7e-6, bn=512, decay=1e-6)
    # train_PFA_cifar10_constraint_all('rAFA_one_layer_512_x4_constraint_all_batch_norm_32_128_128', max_lr=7e-6, bn=512, decay=1e-6)
    # train_PFA_cifar10_constraint_all('rAFA_one_layer_512_x4_constraint_all_batch_norm_32_64_256', max_lr=7e-6, bn=512, decay=1e-6)
    # train_PFA_cifar10_constraint_all('rAFA_one_layer_512_x4_constraint_all_batch_norm_16_64_256', max_lr=7e-6, bn=512, decay=1e-6)
    # train_PFA_cifar10_BP('rAFA_one_layer_128_x4_constraint_all_batch_norm_BP', max_lr=8e-6, bn=128, decay=1e-6)
######################################################################################################################################################

    # train_PFA_cifar100_subsets('rAFA_one_layer_512_x3_subsets_batch_norm_test', 'layer4', max_lr=8.5e-6, bn=512, ranks=[42], decay=1e-6, epochs=60, pct_start=0.2, class_limit=20)
    # train_PFA_cifar100_subsets('rAFA_one_layer_512_x3_subsets_batch_norm', 'layer4', max_lr=8.5e-6, bn=512, ranks=[128], decay=1e-6, epochs=60, pct_start=0.2, class_limit=50)
    # train_PFA_cifar100_subsets('rAFA_one_layer_512_x3_subsets_batch_norm', 'layer4', max_lr=8.5e-6, bn=512, ranks=[16, 100, 256], decay=1e-6, epochs=60, pct_start=0.2, class_limit=100)
    # train_PFA_cifar100_subsets('rAFA_one_layer_512_x3_subsets_batch_norm', 'layer2', max_lr=8.5e-6, bn=512, ranks=[1, 4, 8, 16, 32, 64], decay=1e-6, epochs=60, pct_start=0.2, class_limit=75)
    
    
    # CUDA_VISIBLE_DEVICES=1 python train_cifar_fc_one_cycle.py
    
    