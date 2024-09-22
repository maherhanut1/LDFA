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


from models.optim.AdeMamix import AdEMAMix
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
    'layer4': 10,
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
    
    
    

def train_PFA_cifar100_subsets(session_name, layer, max_lr =1e-4, bn=512, ranks = [None], decay=1e-6, class_limit=100):
    
    device = 'cuda'
    batch_size = 32
    total_classes = class_limit
    num_expirements = 5
    epochs = 100
    session_name = session_name
    dset_name = f'cifar_subsets'
    trainloader, testloader = get_cifar100_loaders(batch_size=batch_size, class_limit=class_limit)
    all_accuracies = dict()
    
    for rank in ranks:
        all_accuracies[rank] = []
        for i in range(num_expirements):
            model = FC_rAFA(input_dim=3*32*32, hidden_dim=bn, num_classes=total_classes)
            model.net[LAYER_IDX[layer]] = rAFALinear(bn, bn, rank=rank, requires_gt=False, update_P=True, update_Q=True) if layer != 'layer4' else rAFALinear(bn,total_classes, rank=min(rank,total_classes), requires_gt=True, update_P=True, update_Q=True)
                
            model.to(device)
            max_lr = max_lr
            tm = TrainingManager(model,
                            trainloader,
                            testloader,
                            optim.Adam,
                            nn.CrossEntropyLoss(),
                            epochs,
                            ExponentialLR,
                            rf"/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset_name}/{session_name}/{class_limit}/{layer}/r_{rank}/exp_{i}",
                            optimizer_params={'lr': max_lr, 'weight_decay': decay, 'amsgrad': True},
                            scheduler_params={'gamma': 0.98},
                            device=device
                            )
            
            val_accuracy = tm.train_model()
            all_accuracies[rank].append(val_accuracy)
            
    import json
    
    
    with open(rf'/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset_name}/{session_name}/{class_limit}/{layer}/accuracies.json', 'w') as f:
        json.dump(all_accuracies, f)
        
        
def train_PFA_cifar100_subsets_BP(session_name, max_lr =1e-4, bn=512, decay=1e-6, class_limit=100):
    
    device = 'cuda'
    batch_size = 32
    total_classes = class_limit
    num_expirements = 10
    epochs = 100
    session_name = session_name
    dset_name = f'cifar_subsets'
    trainloader, testloader = get_cifar100_loaders(batch_size=batch_size, class_limit=class_limit)
    all_accuracies = dict()
    
    all_accuracies['BP'] = []
    for i in range(num_expirements):
        model = FC(input_dim=3*32*32, hidden_dim=bn, num_classes=total_classes)
            
        model.to(device)
        max_lr = max_lr
        tm = TrainingManager(model,
                        trainloader,
                        testloader,
                        optim.Adam,
                        nn.CrossEntropyLoss(),
                        epochs,
                        ExponentialLR,
                        rf"/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset_name}/{session_name}/{class_limit}/BP/exp_{i}",
                        optimizer_params={'lr': max_lr, 'weight_decay': decay, 'amsgrad': True},
                        scheduler_params={'gamma': 0.98},
                        device=device
                        )
        
        val_accuracy = tm.train_model()
        all_accuracies['BP'].append(val_accuracy)
            
    import json
    
    
    with open(rf'/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset_name}/{session_name}/{class_limit}/BP/accuracies.json', 'w') as f:
        json.dump(all_accuracies, f)


def train_PFA_cifar10_exp_decay(session_name, layer, max_lr =1e-4, bn=512, ranks = [None], decay=1e-6, update_p=True):
    
    device = 'cuda'
    batch_size = 32
    total_classes = 10
    epochs = 200
    num_expirements = 10
    session_name = session_name
    dset_name = 'cifar10'
    trainloader, testloader = get_cifar10_loaders(batch_size=batch_size)
    all_accuracies = dict()
    # {'max_lr': max_lr, 'total_steps': epochs*len(trainloader), 'div_factor': 10, 'final_div_factor': 100}
    for i, rank in enumerate(ranks):
        max_lr = max_lr
        all_accuracies[rank] = [] 
        for i in range(num_expirements):
            model = FC_rAFA(input_dim=3*32*32, hidden_dim=bn, num_classes=total_classes)
            model.net[LAYER_IDX[layer]] = rAFALinear(bn, bn, rank=rank, update_P=update_p, update_Q=True, requires_gt=False) if layer != 'layer4' else rAFALinear(bn,total_classes, rank=rank, update_Q=True, update_P=update_p, requires_gt=True)
            model.to(device)
            tm = TrainingManager(model,
                            trainloader,
                            testloader,
                            optim.Adam,
                            nn.CrossEntropyLoss(),
                            epochs,
                            ExponentialLR,
                            rf"/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset_name}/{session_name}/{layer}/r_{rank}/exp_{i}",
                            optimizer_params={'lr': max_lr, 'weight_decay': decay, 'amsgrad': True},
                            scheduler_params={'gamma': 0.98},  #0.97
                            device=device
                            )
            
            val_accuracy = tm.train_model()
            all_accuracies[rank].append(val_accuracy)
            
    import json
    
    
    with open(rf'/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset_name}/{session_name}/{layer}/accuracies.json', 'w') as f:
        json.dump(all_accuracies, f)



        
def train_PFA_cifar10_constraint_all(session_name, ranks, max_lr=8e-5, bn=512, decay=1e-6):
    
    device = 'cuda'
    batch_size = 32
    total_classes = 10
    epochs= 150
    num_expirements = 10
    session_name = session_name
    dset_name = 'cifar10'
    trainloader, testloader = get_cifar10_loaders(batch_size=batch_size)
    all_accuracies = dict()
    
    for rank in ranks:
        all_accuracies[rank] = []
        for i in range(num_expirements):
            model = FC_rAFA(input_dim=3*32*32, hidden_dim=bn, num_classes=total_classes)
            model.net[3] = rAFALinear(bn, bn, rank=rank, requires_gt=False, update_P=True, update_Q=True)
            model.net[6] = rAFALinear(bn, bn, rank=rank, requires_gt=False, update_P=True, update_Q=True)
            model.net[10] = rAFALinear(bn, total_classes, rank=min(total_classes, rank), requires_gt=True, update_P=True, update_Q=True)
            
            model.to(device)
            max_lr = max_lr
            tm = TrainingManager(model,
                            trainloader,
                            testloader,
                            optim.Adam,
                            nn.CrossEntropyLoss(),
                            epochs,
                            ExponentialLR,
                            rf"/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset_name}/{session_name}/constraint_all/r_{rank}/exp_{i}",
                            optimizer_params={'lr': max_lr, 'weight_decay': decay, 'amsgrad':True},
                            scheduler_params={'gamma': 0.98},
                            device=device
                            )
            
            val_accuracy = tm.train_model()
            all_accuracies[rank].append(val_accuracy)
            
    import json
    
    
    with open(rf'/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset_name}/{session_name}/all/accuracies.json', 'w') as f:
        json.dump(all_accuracies, f)
        

def train_PFA_cifar10_BP(session_name, max_lr =8e-6, bn=512, decay=1e-6):
    
    device = 'cuda'
    batch_size = 32
    total_classes = 10
    epochs= 150
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
                            optim.Adam,
                            nn.CrossEntropyLoss(),
                            epochs,
                            ExponentialLR,
                            rf"/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset_name}/{session_name}/all/r_{bn}/exp_{i}",
                            optimizer_params={'lr': max_lr, 'weight_decay': decay},
                            scheduler_params={'gamma': 0.98},
                            device=device
                            )
            
            val_accuracy = tm.train_model()
            all_accuracies[rank].append(val_accuracy)
            
    import json
    
    
    with open(rf'/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset_name}/{session_name}/all/accuracies.json', 'w') as f:
        json.dump(all_accuracies, f)

if __name__ == "__main__":

    ######### 128 x 4 uncomplete tests (params should be near fine for 8 not for lower ....) # optimize per layer per rank
    # train_PFA_cifar10_exp_decay('128x4xmulti_optimization', 'layer2', max_lr= 1.7e-3, bn=128, ranks=[1, 2, 3], decay=5e-4, update_p = True)
    # train_PFA_cifar10_exp_decay('128x4xmulti_optimization', 'layer2', max_lr= 1.7e-3, bn=128, ranks=[4, 5, 6], decay=5e-4, update_p = True)
    # train_PFA_cifar10_exp_decay('128x4xmulti_optimization', 'layer3', max_lr= 1.7e-3, bn=128, ranks=[1, 2, 3], decay=5e-4, update_p = True)
    # train_PFA_cifar10_constraint_all('512x4_no_drop_out_V2_const_all', rank=32, max_lr=4e-4, bn=512, decay=6e-4)
    
    
    # train_PFA_cifar10_exp_decay('512x4_no_drop_out_V2', 'layer3', max_lr= 4e-4, bn=512, ranks=[512, 256], decay=6e-4, update_p = True)
    # train_PFA_cifar10_exp_decay('512x4_no_drop_out_V2', 'layer2', max_lr= 4e-4, bn=512, ranks=[512, 256], decay=6e-4, update_p = True)
    
    
    
    #smaller Nets
    # train_PFA_cifar10_constraint_all('32x4_no_drop_out_V2_no_constraint', rank=32, max_lr=1.5e-3, bn=32, decay=6e-4)
    # train_PFA_cifar10_constraint_all('64x4_no_drop_out_V2_no_constraint', rank=64, max_lr=1.5e-3, bn=64, decay=6e-4) 
    
    # train_PFA_cifar10_exp_decay('64x4_no_drop_out_V2_BP', 'layer2', max_lr= 1.5e-3, bn=64, ranks=[64], decay=5e-4, update_p = True)
    # train_PFA_cifar10_exp_decay('128x4_no_drop_out_V2_BP', 'layer2', max_lr= 1.5e-3, bn=128, ranks=[64], decay=5e-4, update_p = True)
    
    
    # train_PFA_cifar10_BP('512x4_no_drop_out_V2_BP_with_dropout', max_lr=5e-4, bn=512, decay=1e-6)
    
    
    #FInal COnfigs
    
    ############################## 512 NEURONS #####################################################
    
    
    # train_PFA_cifar10_constraint_all('test', ranks=[64], max_lr=5e-4, bn=512, decay=5e-5)
    # train_PFA_cifar10_constraint_all('512x4_no_drop_out_V2_const_all', rank=[32], max_lr=4e-4, bn=512, decay=6e-4)
    
    
    # train_PFA_cifar10_exp_decay('512x4_no_drop_out_V2', 'layer2', max_lr= 4e-4, bn=512, ranks=[1, 2, 3, 4, 5, 6, 8, 10, 16, 32], decay=6e-4, update_p = True)
    # train_PFA_cifar10_exp_decay('test_4e4_6e-4', 'layer3', max_lr= 5e-4, bn=512, ranks=[1, 2, 3, 4, 5, 6, 8, 10, 16, 32][::-1], decay=7e-4, update_p = True)
    # train_PFA_cifar10_exp_decay('512x4_no_drop_out_V2', 'layer4', max_lr= 4e-4, bn=512, ranks=[1, 2, 3, 4, 5, 6, 8, 10], decay=6e-4, update_p = True)
    
    # train_PFA_cifar10_constraint_all('512x4_no_drop_out_V2', rank=[16, 8, 4, 2, 1], max_lr=4e-4, bn=512, decay=5e-4)
    
    
###################################################    # subset ################################################################
    
    # train_PFA_cifar100_subsets('v1', layer='layer4', max_lr=4e-4, bn=512, ranks=[1, 2, 4, 8, 16, 32, 40], decay=5e-5, class_limit=40)
    # train_PFA_cifar100_subsets('v1', layer='layer2', max_lr=4e-4, bn=512, ranks=[1, 2, 4, 8, 16, 32, 64, 100], decay=5e-5, class_limit=100)
    # train_PFA_cifar100_subsets_BP('v1', max_lr=4e-4, bn=512, decay=5e-5, class_limit=40)
    
    

    
    
    
    train_PFA_cifar10_exp_decay('test_4e4_6e-4', 'layer3', max_lr= 5e-4, bn=512, ranks=[1, 2, 3, 4, 5, 6, 8, 10, 16, 32][::-1], decay=4e-4, update_p = True)