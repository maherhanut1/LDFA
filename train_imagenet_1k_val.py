import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import argparse
import os
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR

from models.layers.rAFA_conv import Conv2d as AFAConv


def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training with Validation')
    parser.add_argument('--data', metavar='DIR', default='/home/maherhanut/ImageNet/train',
                        help='path to dataset')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=0.0005, type=float,
                        help='initial learning rate (default: 0.001)')
    args = parser.parse_args()
    args.world_size = 1 # Number of GPUs
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    mp.spawn(main_worker, nprocs=args.world_size, args=(args,))


def replace_conv_layers(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            # Extract parameters from the existing Conv2d layer
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding
            dilation = module.dilation
            groups = module.groups
            bias = module.bias is not None
            padding_mode = module.padding_mode
            
            # Create a new custom conv layer with the same parameters
            new_conv = AFAConv(
                in_channels,
                out_channels,
                kernel_size[0],
                rank=out_channels,
                stride=stride,
                padding=padding,
                dilation=dilation,
                padding_mode=padding_mode,
                groups=groups
            )
            
            # Optionally, copy the weights and bias from the original layer
            with torch.no_grad():
                new_conv.weight = nn.Parameter(module.weight.clone())
                if bias:
                    new_conv.bias = nn.Parameter(module.bias.clone())
            
            # Replace the original Conv2d layer with the new custom layer
            setattr(model, name, new_conv)
        else:
            # Recursively apply to child modules
            replace_conv_layers(module)


def main_worker(rank, args):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)

    torch.cuda.set_device(rank)
    model = models.resnet50()
    replace_conv_layers(model)
    model = torch.load('RAF_model_imagenet_no_const.pt')
    # torch.save(model, 'RAF_model_imagenet_no_const.pt')
    model.cuda(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(rank)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Initialize the learning rate scheduler
    scheduler = ExponentialLR(optimizer, gamma=0.985)

    # Data loading code
    traindir = os.path.join(args.data)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Define transforms for training and validation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # Load the full dataset
    full_dataset = datasets.ImageFolder(traindir, transform=train_transform)

    # Split the dataset into training and validation sets
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.1 * dataset_size))  # 10% for validation

    np.random.seed(42)  # Ensure reproducibility
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    val_dataset.dataset.transform = val_transform  # Apply validation transforms

    # Create samplers for distributed training
    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    #     train_dataset, num_replicas=args.world_size, rank=rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=args.world_size, rank=rank, shuffle=False)

    # Create data loaders
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size // args.world_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size // args.world_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    validate(val_loader, model, criterion, rank)

def train(train_loader, model, criterion, optimizer, epoch, rank):
    model.train()
    for i, (images, target) in enumerate(train_loader):
        images = images.cuda(rank, non_blocking=True)
        target = target.cuda(rank, non_blocking=True)

        # Compute output
        output = model(images)
        loss = criterion(output, target)

        # Compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0 and rank == 0:
            print(f"Epoch [{epoch}][{i}/{len(train_loader)}]\t Loss {loss.item():.4f}")

def validate(val_loader, model, criterion, rank):
    model.eval()
    correct = torch.tensor(0).cuda(rank)
    total = torch.tensor(0).cuda(rank)
    with torch.no_grad():
        for images, target in val_loader:
            images = images.cuda(rank, non_blocking=True)
            target = target.cuda(rank, non_blocking=True)

            # Compute output
            output = model(images)
            loss = criterion(output, target)

            # Measure accuracy
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum()

    # Reduce across all processes
    dist.reduce(correct, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(total, dst=0, op=dist.ReduceOp.SUM)

    if rank == 0:
        accuracy = correct.float() / total.float()
        print(f'Validation Accuracy: {accuracy.item() * 100:.2f}%')
    
    
    
    # example_input = torch.randn(1, 3, 224, 224)
    # traced_model = torch.jit.trace(model, example_input)
    # torch.jit.save(traced_model, "resnet50_RAF_ratio_2.pt")
    # exit(0)

if __name__ == '__main__':
    main()
