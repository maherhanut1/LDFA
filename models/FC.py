import torch.nn as nn
import torch
from models.layers.rAFA_linear import Linear as rAFA_Linear
from models.layers.FA_linear import Linear as FA_linear

    
class FC(nn.Module):
    

    def __init__(self, input_dim = 1024*3, hidden_dim = 512, num_classes: int = 10, device='cuda') -> None:
        super().__init__()
        
        self.device = device
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 nn.BatchNorm1d(hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, hidden_dim),
                                  nn.BatchNorm1d(hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, hidden_dim),
                                  nn.BatchNorm1d(hidden_dim),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.0),
                                  nn.Linear(hidden_dim, num_classes))

    def forward(self, x: torch.Tensor, gt = None) -> torch.Tensor:
        b, *_ = x.shape
        x = x.view(b, -1)
        for layer in self.net:
            x = layer(x, gt) if getattr(layer, 'requires_gt', False) else layer(x)
        return x, None
    

class FC_rAFA(nn.Module):
    

    def __init__(self, input_dim = 1024*3, hidden_dim = 512, num_classes: int = 10, device='cuda') -> None:
        super().__init__()
        
        self.device = device
        self.net = nn.Sequential(rAFA_Linear(input_dim, hidden_dim, update_P=False, update_Q=True, requires_gt=False, rank=hidden_dim),
                                 nn.BatchNorm1d(hidden_dim),
                                  nn.ReLU(),
                                  rAFA_Linear(hidden_dim, hidden_dim, update_P=True, update_Q=True, requires_gt=False, rank=hidden_dim),
                                  nn.BatchNorm1d(hidden_dim),
                                  nn.ReLU(),
                                  rAFA_Linear(hidden_dim, hidden_dim, update_P=True, update_Q=True, requires_gt=False, rank=hidden_dim),
                                  nn.BatchNorm1d(hidden_dim),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.0),
                                  rAFA_Linear(hidden_dim, num_classes, update_P=True, update_Q=True, requires_gt=True, rank=min(hidden_dim, num_classes)))

    def forward(self, x: torch.Tensor, gt = None) -> torch.Tensor:
        b, *_ = x.shape
        x = x.view(b, -1)
        for layer in self.net:
            x = layer(x, gt) if getattr(layer, 'requires_gt', False) else layer(x)
        return x, None
    
    
    

    
