import torch.nn as nn
import torch
from models.layers.rAFA_linear import Linear as rAFA_Linear

    
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
                                  nn.Dropout(p=0.4),
                                  nn.Linear(hidden_dim, num_classes))

    def forward(self, x: torch.Tensor, gt = None) -> torch.Tensor:
        b, *_ = x.shape
        x = x.view(b, -1)
        for layer in self.net:
            x = layer(x, gt) if getattr(layer, 'requires_gt', False) else layer(x)
        return x, None
    

class FC_rAFA(nn.Module):
    
    def __init__(self, input_dim = 1024*3, hidden_dim = 512, num_classes: int = 10, backward_constraint=None, device='cuda') -> None:
        super().__init__()
        
        self.device = device
        if backward_constraint is None:
            backward_constraint = hidden_dim
            
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 nn.BatchNorm1d(hidden_dim),
                                 nn.ReLU(),
                                 rAFA_Linear(hidden_dim, hidden_dim, rank=backward_constraint),
                                 nn.BatchNorm1d(hidden_dim),
                                 nn.ReLU(),
                                 rAFA_Linear(hidden_dim, hidden_dim, rank=backward_constraint),
                                 nn.BatchNorm1d(hidden_dim),
                                 nn.ReLU(),
                                 rAFA_Linear(hidden_dim, num_classes, rank=backward_constraint))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, *_ = x.shape
        return self.net(x.view(b, -1)), None
    
