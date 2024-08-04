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
from models.layers.rAFA_linear import Linear as rAFALinear
from utils.model_trainer import TrainingManager

import torchvision.datasets as datasets

# LAYER_IDX = {
#     'layer1': 0,
#     'layer2': 2,
#     'layer3': 4,
#     'layer4': 6,
#     'layer5': 8,
#     'layer6': 10
# }

LAYER_IDX = {
    'layer1': 0,
    'layer2': 3,
    'layer3': 6,
    'layer4': 9,
    # 'layer5': 8,
    # 'layer6': 10
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

def train_PFA_cifar100(session_name, layer, max_lr =1e-4, bn=512, ranks = [None], decay=1e-6, pct_start=0.3, epochs=120, class_limit=1000):
    
    device = 'cuda'
    batch_size = 64
    total_classes = class_limit
    num_expirements = 10
    session_name = session_name
    dset_name = f'cifar{class_limit}'
    trainloader, testloader = get_cifar100_loaders(batch_size=batch_size, class_limit=total_classes)
    all_accuracies = dict()
    
    for rank in ranks:
        all_accuracies[rank] = []
        for i in range(num_expirements):
            model = FC3(input_dim=3*32*32, hidden_dim=bn, num_classes=total_classes)
            model.net[LAYER_IDX[layer]] = rAFALinear(bn, bn, rank=rank) if layer != 'layer3' else rAFALinear(bn,total_classes, rank=rank)
                
            model.to(device)
            max_lr = max_lr
            tm = TrainingManager(model,
                            trainloader,
                            testloader,
                            optim.RMSprop,
                            nn.CrossEntropyLoss(),
                            epochs,
                            OneCycleLR,
                            rf"/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset_name}/{session_name}/{layer}/r_{rank}/exp_{i}",
                            optimizer_params={'lr': max_lr, 'weight_decay': decay},
                            scheduler_params={'max_lr': max_lr, 'total_steps': epochs*len(trainloader), 'div_factor': 10, 'final_div_factor': 200, 'pct_start':pct_start},
                            device=device
                            )
            
            val_accuracy = tm.train_model()
            all_accuracies[rank].append(val_accuracy)
            
    import json
    
    

def train_PFA_cifar100_subsets(session_name, layer, max_lr =1e-4, bn=512, ranks = [None], decay=1e-6, pct_start=0.3, epochs=120, class_limit=100):
    
    device = 'cuda'
    batch_size = 64
    total_classes = class_limit
    num_expirements = 5
    session_name = session_name
    dset_name = f'cifar_subsets'
    trainloader, testloader = get_cifar100_loaders(batch_size=batch_size, class_limit=class_limit)
    all_accuracies = dict()
    
    for rank in ranks:
        all_accuracies[rank] = []
        for i in range(num_expirements):
            model = FC(input_dim=3*32*32, hidden_dim=bn, num_classes=total_classes)
            model.net[LAYER_IDX[layer]] = rAFALinear(bn, bn, rank=rank) if layer != 'layer4' else rAFALinear(bn,total_classes, rank=rank)
                
            model.to(device)
            max_lr = max_lr
            tm = TrainingManager(model,
                            trainloader,
                            testloader,
                            optim.RMSprop,
                            nn.CrossEntropyLoss(),
                            epochs,
                            OneCycleLR,
                            rf"/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset_name}/{session_name}/{layer}_{class_limit}/r_{rank}/exp_{i}",
                            optimizer_params={'lr': max_lr, 'weight_decay': decay},
                            scheduler_params={'max_lr': max_lr, 'total_steps': epochs*len(trainloader), 'div_factor': 10, 'final_div_factor': 100, 'pct_start':pct_start},
                            device=device
                            )
            
            val_accuracy = tm.train_model()
            all_accuracies[rank].append(val_accuracy)
            
    import json
    
    
    with open(rf'/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset_name}/{session_name}/{layer}_{class_limit}/accuracies.json', 'w') as f:
        json.dump(all_accuracies, f)


def train_PFA_cifar10(session_name, layer, max_lr =1e-4, bn=512, ranks = [None], decay=1e-6):
    
    device = 'cpu'
    batch_size = 64
    total_classes = 10
    epochs= 50
    num_expirements = 10
    session_name = session_name
    dset_name = 'cifar10'
    trainloader, testloader = get_cifar10_loaders(batch_size=batch_size)
    all_accuracies = dict()
    # {'max_lr': max_lr, 'total_steps': epochs*len(trainloader), 'div_factor': 10, 'final_div_factor': 100}
    for rank in ranks:
        all_accuracies[rank] = []
        for i in range(num_expirements):
            model = FC(input_dim=3*32*32, hidden_dim=bn, num_classes=total_classes)
            model.net[LAYER_IDX[layer]] = rAFALinear(bn, bn, rank=rank) if layer != 'layer4' else rAFALinear(bn,total_classes, rank=rank, update_Q=True, update_P=True)
            model.to(device)
            max_lr = max_lr
            tm = TrainingManager(model,
                            trainloader,
                            testloader,
                            optim.RMSprop,
                            nn.CrossEntropyLoss(),
                            epochs,
                            OneCycleLR,
                            rf"/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset_name}/{session_name}/{layer}/r_{rank}/exp_{i}",
                            optimizer_params={'lr': max_lr, 'weight_decay': decay},
                            scheduler_params={'max_lr': max_lr, 'total_steps': epochs*len(trainloader), 'div_factor': 10, 'final_div_factor': 100, 'pct_start': 0.15},
                            device=device
                            )
            
            val_accuracy = tm.train_model()
            all_accuracies[rank].append(val_accuracy)
            
    import json
    
    
    with open(rf'/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset_name}/{session_name}/{layer}/accuracies.json', 'w') as f:
        json.dump(all_accuracies, f)



def train_PFA_cifar10_no_constraint(session_name, max_lr =8e-6, bn=512, decay=1e-6):
    
    device = 'cuda'
    batch_size = 64
    total_classes = 10
    epochs= 50
    num_expirements = 10
    session_name = session_name
    dset_name = 'cifar10'
    trainloader, testloader = get_cifar10_loaders(batch_size=batch_size)
    all_accuracies = dict()
    
    for rank in [bn]:
        all_accuracies[rank] = []
        for i in range(num_expirements):
            model = FC_rAFA(input_dim=3*32*32, hidden_dim=bn, num_classes=total_classes, backward_constraint=bn)
            
            
            model.to(device)
            max_lr = max_lr
            tm = TrainingManager(model,
                            trainloader,
                            testloader,
                            optim.RMSprop,
                            nn.CrossEntropyLoss(),
                            epochs,
                            OneCycleLR,
                            rf"/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset_name}/{session_name}/all/r_{bn}/exp_{i}",
                            optimizer_params={'lr': max_lr, 'weight_decay': decay},
                            scheduler_params={'max_lr': max_lr, 'total_steps': epochs*len(trainloader), 'div_factor': 10, 'final_div_factor': 100, 'pct_start': 0.15},
                            device=device
                            )
            
            val_accuracy = tm.train_model()
            all_accuracies[rank].append(val_accuracy)
            
    import json
    
    
    with open(rf'/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset_name}/{session_name}/all/accuracies.json', 'w') as f:
        json.dump(all_accuracies, f)
        
def train_PFA_cifar10_constraint_all(session_name, max_lr=8e-6, bn=512, decay=1e-6):
    
    device = 'cuda'
    batch_size = 64
    total_classes = 10
    epochs= 50
    num_expirements = 10
    session_name = session_name
    dset_name = 'cifar10'
    trainloader, testloader = get_cifar10_loaders(batch_size=batch_size)
    all_accuracies = dict()
    
    for rank in [bn]:
        all_accuracies[rank] = []
        for i in range(num_expirements):
            model = FC(input_dim=3*32*32, hidden_dim=bn, num_classes=total_classes)
            model.net[3] = rAFALinear(bn, bn, rank=32)
            model.net[6] = rAFALinear(bn, bn, rank=64)
            model.net[9] = rAFALinear(bn, total_classes, rank=128)
            
            model.to(device)
            max_lr = max_lr
            tm = TrainingManager(model,
                            trainloader,
                            testloader,
                            optim.RMSprop,
                            nn.CrossEntropyLoss(),
                            epochs,
                            OneCycleLR,
                            rf"/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset_name}/{session_name}/all/r_{bn}/exp_{i}",
                            optimizer_params={'lr': max_lr, 'weight_decay': decay},
                            scheduler_params={'max_lr': max_lr, 'total_steps': epochs*len(trainloader), 'div_factor': 10, 'final_div_factor': 100, 'pct_start': 0.15},
                            device=device
                            )
            
            val_accuracy = tm.train_model()
            all_accuracies[rank].append(val_accuracy)
            
    import json
    
    
    with open(rf'/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset_name}/{session_name}/all/accuracies.json', 'w') as f:
        json.dump(all_accuracies, f)
        

def train_PFA_cifar10_BP(session_name, max_lr =8e-6, bn=512, decay=1e-6):
    
    device = 'cuda'
    batch_size = 64
    total_classes = 10
    epochs= 50
    num_expirements = 10
    session_name = session_name
    dset_name = 'cifar10'
    trainloader, testloader = get_cifar10_loaders(batch_size=batch_size)
    all_accuracies = dict()
    
    for rank in [bn]:
        all_accuracies[rank] = []
        for i in range(num_expirements):
            model = FC(input_dim=3*32*32, hidden_dim=bn, num_classes=total_classes)
                
            model.to(device)
            max_lr = max_lr
            tm = TrainingManager(model,
                            trainloader,
                            testloader,
                            optim.RMSprop,
                            nn.CrossEntropyLoss(),
                            epochs,
                            OneCycleLR,
                            rf"/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset_name}/{session_name}/all/r_{bn}/exp_{i}",
                            optimizer_params={'lr': max_lr, 'weight_decay': decay},
                            scheduler_params={'max_lr': max_lr, 'total_steps': epochs*len(trainloader), 'div_factor': 10, 'final_div_factor': 100, 'pct_start': 0.15},
                            device=device
                            )
            
            val_accuracy = tm.train_model()
            all_accuracies[rank].append(val_accuracy)
            
    import json
    
    
    with open(rf'/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset_name}/{session_name}/all/accuracies.json', 'w') as f:
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
   train_PFA_cifar10('update_P_test', 'layer4', max_lr=4e-6, bn=512, ranks=[128], decay=1e-6)
    # train_PFA_cifar10('rAFA_one_layer_128_x4_batch_norm', 'layer2', max_lr=8e-6, bn=128, ranks=[64, 32, 20, 16, 8, 4, 2, 1], decay=1e-6)
######################################################################################################################################################

########################################### train on cifar10 512 neurons x 4 layers x constrain all layers ranks, BP, and no constraints ########################################### 
    # train_PFA_cifar10_no_constraint('rAFA_one_layer_512_x4_no_constraint_batch_norm', max_lr=8e-6, bn=512, decay=1e-6)
    # train_PFA_cifar10_constraint_all('rAFA_one_layer_512_x4_constraint_all_batch_norm_32_64_128', max_lr=7e-6, bn=512, decay=1e-6)
    # train_PFA_cifar10_constraint_all('rAFA_one_layer_512_x4_constraint_all_batch_norm_64_64_128', max_lr=7e-6, bn=512, decay=1e-6)
    # train_PFA_cifar10_constraint_all('rAFA_one_layer_512_x4_constraint_all_batch_norm_16_96_128', max_lr=7e-6, bn=512, decay=1e-6)
    
    # train_PFA_cifar10_constraint_all('rAFA_one_layer_512_x4_constraint_all_batch_norm_64_128_256', max_lr=7e-6, bn=512, decay=1e-6)
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
    
    # train_PFA_cifar10_BP('rAFA_one_layer_128_x4_constraint_all_batch_norm_BP', max_lr=8e-6, bn=128, decay=1e-6)
    # CUDA_VISIBLE_DEVICES=1 python train_cifar_fc_one_cycle.py
    
    