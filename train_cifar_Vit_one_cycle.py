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
from torch.utils.tensorboard import SummaryWriter

from models.ViT import ViT
from models.RAF_ViT import RafViT
from models.FC import FC_rAFA
from models.layers.rAFA_linear_att import Linear as rAFALinear
# from utils.model_trainer import TrainingManager


from models.optim.AdeMamix import AdEMAMix
import torchvision.datasets as datasets

# Configuration for paths - can be overridden by environment variables
DATA_DIR = os.getenv('LDFA_DATA_DIR', './data')
ARTIFACTS_DIR = os.getenv('LDFA_ARTIFACTS_DIR', './artifacts')

def get_mnist_loaders(batch_size=32):
    
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # MNIST is grayscale, single channel
])

    # Downloading/loading the MNIST training dataset
    train_dataset = datasets.MNIST(root=DATA_DIR, train=True, transform=transform, download=True)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Downloading/loading the MNIST test dataset
    test_dataset = datasets.MNIST(root=DATA_DIR, train=False, transform=transform, download=True)
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

    trainset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_test)
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

    trainset = datasets.CIFAR100(root=DATA_DIR, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR100(root=DATA_DIR, train=False, download=True, transform=transform_test)
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
    num_expirements = 1
    session_name = session_name
    dset_name = 'cifar10'
    trainloader, testloader = get_cifar10_loaders(batch_size=batch_size)
    all_accuracies = dict()
    
    for rank in ranks:
        all_accuracies[rank] = []
        for i in range(num_expirements):
            model = class_type(3, 10, 32, 8, hidden=384, mlp_hidden=384, dropout=0.1)
            # model = FC_rAFA()
                
            model.to(device)
            max_lr = max_lr 
            
            # Setup optimizers and loss
            # Separate P parameters from other parameters
            p_params = []
            other_params = []
            
            for name, param in model.named_parameters():
                if 'P' in name:
                    p_params.append(param)
                else:
                    other_params.append(param)
            
            # Create separate optimizers
            optimizer = optim.AdamW(other_params, lr=max_lr, weight_decay=decay)
            p_optimizer = optim.AdamW(p_params, lr=1.5*max_lr, weight_decay=5e-4)
            
            # Log parameter counts for debugging
            total_p_params = sum(p.numel() for p in p_params)
            total_other_params = sum(p.numel() for p in other_params)
            print(f"P parameters: {total_p_params}, Other parameters: {total_other_params}")
            
            criterion = nn.CrossEntropyLoss()
            scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=epochs*len(trainloader), pct_start=0.15, div_factor=15)
            p_scheduler = OneCycleLR(p_optimizer, max_lr=1.5*max_lr, total_steps=epochs*len(trainloader), pct_start=0.15, div_factor=10)
            
            # Setup logging
            session_path = Path(ARTIFACTS_DIR) / dset_name / session_name / 'RAF' / f'r_{rank}' / f'exp_{i}'
            logdir = session_path / 'tensorboard_logs'
            logdir.mkdir(exist_ok=True, parents=True)
            writer = SummaryWriter(log_dir=logdir)
            model_path = session_path / 'checkpoints'
            model_path.mkdir(exist_ok=True, parents=True)
            
            # Training loop
            best_accuracy = 0
            for epoch in range(epochs):
                # Training phase
                model.train()
                running_loss = 0.0
                running_acc = 0.0
                
                for batch_idx, (inputs, labels) in enumerate(tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}')):
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    p_optimizer.zero_grad()
                    
                    outputs, regularization_losses = model(inputs, labels)
                    loss = criterion(outputs, labels)
                    
                    if regularization_losses is not None:
                        loss += regularization_losses
                    
                    loss.backward()
                    optimizer.step()
                    p_optimizer.step()
                    running_loss += loss.item()
                    
                    pred_labels = torch.argmax(outputs, dim=1)
                    accuracy = (pred_labels == labels).float().mean().item()
                    running_acc += accuracy
                    
                    if batch_idx % 100 == 0:
                        writer.add_scalar('Loss/train', loss.item(), epoch * len(trainloader) + batch_idx)
                        writer.add_scalar('Accuracy/train', accuracy, epoch * len(trainloader) + batch_idx)
                    
                    del inputs, outputs, labels
                
                # Update learning rate
                    scheduler.step()
                    p_scheduler.step()

                # p_scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                current_p_lr = p_optimizer.param_groups[0]['lr']
                writer.add_scalar('LR', current_lr, epoch)
                writer.add_scalar('LR_P', current_p_lr, epoch)
                
                # Calculate epoch metrics
                epoch_loss = running_loss / len(trainloader)
                epoch_accuracy = running_acc / len(trainloader)
                
                # Validation phase
                model.eval()
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs, _ = model(inputs, labels)
                        pred_labels = torch.argmax(outputs, dim=1)
                        val_correct += (pred_labels == labels).sum().item()
                        val_total += labels.size(0)
                
                val_accuracy = val_correct / val_total
                
                # Save best model
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    torch.save(model.state_dict(), model_path / "best.pth")
                
                # Log metrics
                writer.add_scalar('Accuracy/val', val_accuracy, epoch)
                writer.add_scalar('Loss/epoch', epoch_loss, epoch)
                writer.add_scalar('Accuracy/epoch', epoch_accuracy, epoch)
                
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")
            
            # Save final model
            torch.save(model.state_dict(), model_path / "final.pth")
            print(f"Final model saved checkpoint at {model_path}")
            
            # Close TensorBoard writer
            writer.close()
            
            # Store final accuracy
            val_accuracy = max(val_accuracy, best_accuracy)
            all_accuracies[rank].append(val_accuracy)
            
    import json
    
    # Save accuracies to artifacts directory
    accuracies_path = Path(ARTIFACTS_DIR) / dset_name / session_name / 'RAF' / 'accuracies.json'
    accuracies_path.parent.mkdir(exist_ok=True, parents=True)
    with open(accuracies_path, 'w') as f:
        json.dump(all_accuracies, f)

if __name__ == "__main__":
    
    
    #set 1
    lrs_set2 = [3e-4]
    # lrs_set1 = [5e-5, 1e-5]
    decays = [1e-4]
    
    
    train_PFA_cifar10_gen(RafViT, f'RAFVIT_rank_16', max_lr=8e-4, bn=512, decay=5e-5, ranks=[10])