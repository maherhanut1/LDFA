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


# def train_PFA_cifar10(session_name, layer, max_lr =1e-4, bn=512, ranks = [None], decay=1e-6):
    
#     device = 'cuda'
#     batch_size = 64
#     total_classes = 10
#     epochs= 80
#     num_expirements = 10
#     session_name = session_name
#     dset_name = 'cifar10'
#     trainloader, testloader = get_cifar10_loaders(batch_size=batch_size)
#     all_accuracies = dict()
#     # {'max_lr': max_lr, 'total_steps': epochs*len(trainloader), 'div_factor': 10, 'final_div_factor': 100}
#     for rank in ranks:
#         all_accuracies[rank] = []
#         for i in range(num_expirements):
#             model = FC(input_dim=3*32*32, hidden_dim=bn, num_classes=total_classes)
#             model.net[LAYER_IDX[layer]] = rAFALinear(bn, bn, rank=rank, update_P=True, update_Q=True, requires_gt=False) if layer != 'layer4' else rAFALinear(bn,total_classes, rank=rank, update_Q=True, update_P=True, requires_gt=True)
#             model.to(device)
#             max_lr = max_lr
#             tm = TrainingManager(model,
#                             trainloader,
#                             testloader,
#                             optim.RMSprop,
#                             nn.CrossEntropyLoss(),
#                             epochs,
#                             OneCycleLR,
#                             rf"/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset_name}/{session_name}/{layer}/r_{rank}/exp_{i}",
#                             optimizer_params={'lr': max_lr, 'weight_decay': decay},
#                             scheduler_params={'max_lr': max_lr, 'total_steps': epochs*len(trainloader), 'div_factor': 10, 'final_div_factor': 1e5, 'pct_start': 0.4},
#                             device=device
#                             )
            
#             val_accuracy = tm.train_model()
#             all_accuracies[rank].append(val_accuracy)
            
#     import json
    
    
#     with open(rf'/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset_name}/{session_name}/{layer}/accuracies.json', 'w') as f:
#         json.dump(all_accuracies, f)



def train_PFA_cifar10(session_name, layer, max_lr =1e-4, bn=512, ranks = [None], decay=1e-6):
    
    device = 'cpu'
    batch_size = 64
    total_classes = 10
    epochs= 60
    num_expirements = 10
    session_name = session_name
    dset_name = 'cifar10'
    trainloader, testloader = get_cifar10_loaders(batch_size=batch_size)
    all_accuracies = dict()
    # {'max_lr': max_lr, 'total_steps': epochs*len(trainloader), 'div_factor': 10, 'final_div_factor': 100}
    for rank in ranks:
        all_accuracies[rank] = []
        for i in range(num_expirements):
            model = FC_rAFA(input_dim=3*32*32, hidden_dim=bn, num_classes=total_classes)
            model.net[LAYER_IDX[layer]] = rAFALinear(bn, bn, rank=rank, update_P=True, update_Q=True, requires_gt=False) if layer != 'layer4' else rAFALinear(bn,total_classes, rank=rank, update_Q=True, update_P=True, requires_gt=True)
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
                            scheduler_params={'max_lr': max_lr, 'total_steps': epochs, 'div_factor': 15, 'final_div_factor': 5e5, 'pct_start': 0.12, 'anneal_strategy': 'cos'},
                            device=device
                            )
            
            val_accuracy = tm.train_model()
            all_accuracies[rank].append(val_accuracy)
            
    import json
    
    
    with open(rf'/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset_name}/{session_name}/{layer}/accuracies.json', 'w') as f:
        json.dump(all_accuracies, f)
        
        

def train_PFA_cifar10_exp_decay(session_name, layer, max_lr =1e-4, bn=512, ranks = [None], decay=1e-6, update_p=True):
    
    device = 'cuda'
    batch_size = 32
    total_classes = 10
    epochs = 150
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
                            scheduler_params={'gamma': 0.97},  #0.97
                            device=device
                            )
            
            val_accuracy = tm.train_model()
            all_accuracies[rank].append(val_accuracy)
            
    import json
    
    
    with open(rf'/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset_name}/{session_name}/{layer}/accuracies.json', 'w') as f:
        json.dump(all_accuracies, f)



        
def train_PFA_cifar10_constraint_all(session_name, rank, max_lr=8e-5, bn=512, decay=1e-6):
    
    device = 'cuda'
    batch_size = 64
    total_classes = 10
    epochs= 150
    num_expirements = 10
    session_name = session_name
    dset_name = 'cifar10'
    trainloader, testloader = get_cifar10_loaders(batch_size=batch_size)
    all_accuracies = dict()
    
    for rank in [rank]:
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
                            rf"/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset_name}/{session_name}/all_{rank}/r_{rank}/exp_{i}",
                            optimizer_params={'lr': max_lr, 'weight_decay': decay},
                            scheduler_params={'gamma': 0.96},
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
                            scheduler_params={'gamma': 0.97},
                            device=device
                            )
            
            val_accuracy = tm.train_model()
            all_accuracies[rank].append(val_accuracy)
            
    import json
    
    
    with open(rf'/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset_name}/{session_name}/all/accuracies.json', 'w') as f:
        json.dump(all_accuracies, f)

if __name__ == "__main__":

    ranks = [32, 16, 8, 4, 2, 1][::-1]






    ######### 128 x 4 uncomplete tests (params should be near fine for 8 not for lower ....) # optimize per layer per rank
    # train_PFA_cifar10_exp_decay('128x4xmulti_optimization', 'layer2', max_lr= 1.7e-3, bn=128, ranks=[1, 2, 3], decay=5e-4, update_p = True)
    # train_PFA_cifar10_exp_decay('128x4xmulti_optimization', 'layer2', max_lr= 1.7e-3, bn=128, ranks=[4, 5, 6], decay=5e-4, update_p = True)
    # train_PFA_cifar10_exp_decay('128x4xmulti_optimization', 'layer3', max_lr= 1.7e-3, bn=128, ranks=[1, 2, 3], decay=5e-4, update_p = True)
    
    #6e04 decay and 5e-4 lr for no_dropouts_v2
    train_PFA_cifar10_exp_decay('512x4_no_drop_out_V2_test_normalzing_max_std', 'layer3', max_lr= 5e-4, bn=512, ranks=[32, 16, 8, 4], decay=3e-4, update_p = True)
    # train_PFA_cifar10_BP('512x4_no_drop_out_V2_BP_with_dropout', max_lr=5e-4, bn=512, decay=1e-6)
    
    # train_PFA_cifar10_exp_decay('128x4xmulti_optimization', 'layer2', max_lr= 1.7e-3, bn=128, ranks=[1, 2, 3, 4, 5, 6], decay=5e-4, update_p = True)
    # print('5e-4')
    # train_PFA_cifar10_exp_decay('128x4xmulti_optimization', 'layer2', max_lr= 1.7e-3, bn=128, ranks=[8], decay=5e-4, update_p = True)
