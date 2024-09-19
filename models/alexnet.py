import torch.nn as nn
import torch

from models.layers.rAFA_conv import Conv2d as AFAConv

    
class AlexNet_cifar(nn.Module):
    

    def __init__(self, input_channels = 1, bn = 32, kernel_size=9 ,num_classes: int = 10, dropout: float = 0.5, device='cuda') -> None:
        super().__init__()
        
        self.device = device
        
        self.retina = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, bn, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(bn),
            nn.ReLU(),
        )
    
        self.vvs = nn.Sequential(
            nn.Conv2d(bn, 32, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32 * 32 * 32, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x: torch.Tensor, gt=None) -> torch.Tensor:
        x = self.retina(x)
        x = self.vvs(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, None
    
    

class AlexNet_AFA(nn.Module):
    

    def __init__(self, input_channels = 1, bn = 32, kernel_size=9 ,num_classes: int = 10, dropout: float = 0.5, device='cuda', constrain=32, update_p=True, update_q=True) -> None:
        super().__init__()
        
        self.device = device
        
        self.retina = nn.Sequential(
            AFAConv(input_channels, 32, kernel_size=kernel_size, padding=kernel_size//2, rank=constrain, update_q=update_q, update_p=update_p),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            AFAConv(32, bn, kernel_size=kernel_size, padding=kernel_size//2, rank=constrain, update_p=update_p, update_q=update_q),
            nn.BatchNorm2d(bn),
            nn.ReLU(),
        )
    
        self.vvs = nn.Sequential(
            AFAConv(bn, 32, kernel_size=kernel_size, padding=kernel_size//2, rank=constrain, update_p=update_p, update_q=update_q),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            AFAConv(32, 32, kernel_size=kernel_size, padding=kernel_size//2, rank=constrain, update_p=update_p, update_q=update_q),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            AFAConv(32, 32, kernel_size=kernel_size, padding=kernel_size//2, rank=constrain, update_p=update_p, update_q=update_q),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        
        self.classifier = nn.Sequential(
            nn.Linear(32 * 32 * 32, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x: torch.Tensor, gt=None) -> torch.Tensor:
        x = self.retina(x)
        x = self.vvs(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, None