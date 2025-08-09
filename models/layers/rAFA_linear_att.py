import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch import autograd


class LinearGrad(autograd.Function):
    """
    Autograd Function that Does a backward pass using the B matrix of the layer
    """
    @staticmethod
    # Same as reference linear function, but with additional weight tensor for backward
    def forward(context, input, weight, P, Q, bias=None, bias_backward=None, gt=None):
        
        output = input @ (weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        
        context.save_for_backward(input, weight, P, Q, bias, bias_backward, output, gt)
        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, P, Q, bias, bias_backward, output, gt = context.saved_tensors
        grad_input = grad_weight = grad_Q = grad_P = grad_bias = grad_bias_backward = grad_input_intermediate = None
        # Gradient input
        
        if context.needs_input_grad[0]:
            # Use the B matrix to compute the gradient
            grad_input_intermediate = grad_output @ (P)
            grad_input = grad_input_intermediate @ (Q)
            # grad_input = grad_output @ (weight)
            # grad_input_optimal = grad_input_optimal.reshape(-1, grad_input_optimal.shape[-1])
            
        # Gradient weights
        if context.needs_input_grad[1]:
            grad_output = grad_output.reshape(-1, grad_output.shape[-1])
            input = input.view(-1, input.shape[-1])
            grad_weight = grad_output.t() @ (input)
        
        if context.needs_input_grad[2]:
            # grad_P = grad_output.t() @ (input @ Q.t())
            grad_P = grad_weight @ Q.t() #TODO ADD IF STATEMENT TO USE THE MORE EFFICIENT ONE DEPENDING ON DIMENSIONS
              
        if grad_input_intermediate is not None and context.needs_input_grad[3]:
            # grad_Q = grad_input_intermediate.view(-1, grad_input_intermediate.shape[-1]).t() @ (input)
            grad_Q = P.t() @ grad_weight
        
            
        # Gradient bias
        if bias is not None and context.needs_input_grad[4]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_P, grad_Q, grad_bias, grad_bias_backward, None



class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True, layer_config: dict = None, update_P = True, update_Q = True, requires_gt = False) -> None:
        super(Linear, self).__init__(in_features, out_features, bias)
        self.layer_config = layer_config

        if self.layer_config is None:
            self.layer_config = {
                "type": "fa"
            }

        if "options" not in self.layer_config:
            self.layer_config["options"] = {
                "gradient_clip": False,
                "init": "kaiming"
            }
            
        self.options = self.layer_config["options"]
        self.type = self.layer_config["type"]
        self.init = self.options["init"]
        self.requires_gt = True if update_P and requires_gt else False
        self.rank = rank
        self.Q = nn.Parameter(torch.Tensor(self.rank, in_features), requires_grad=update_Q)
        self.P = nn.Parameter(torch.Tensor(out_features, self.rank), requires_grad=update_P)
        
        if self.bias is not None:
            self.bias_backward = nn.Parameter(torch.Tensor(self.bias.size()), requires_grad=False)
        else:
            self.register_parameter("bias", None)
            self.bias_backward = None

        self.init_parameters()
        # self.register_forward_pre_hook(self.normalize_P)
        self.register_full_backward_hook(self.gradient_clip)
        
    def init_svd_approx(self):
        """Initialize P and Q such that QP approximates the current weight matrix W"""
        # Compute SVD of W: W = U @ S @ V^T
        # W shape: (out_features, in_features)
        U, S, Vt = torch.linalg.svd(self.weight.data, full_matrices=False)
        
        # Take top-k singular values and vectors (k = rank)
        k = min(self.rank, len(S))
        U_k = U[:, :k]  # (out_features, k)
        S_k = S[:k]     # (k,)
        Vt_k = Vt[:k, :]  # (k, in_features)
        
        # Initialize P and Q such that QP ≈ W
        # P = U_k * sqrt(S_k), Q = sqrt(S_k) * Vt_k
        sqrt_S_k = torch.sqrt(S_k)
        self.P.data = U_k * sqrt_S_k.unsqueeze(0)  # (out_features, k)
        self.Q.data = (sqrt_S_k.unsqueeze(1) * Vt_k)  # (k, in_features)
        
    def init_parameters(self) -> None:
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
        # Xavier initialization
        if self.init == "xavier":
            nn.init.xavier_uniform_(self.weight)
            nn.init.xavier_uniform_(self.P)
            nn.init.xavier_uniform_(self.Q)
            
            # Scaling factor is the standard deviation of xavier init.
            self.scaling_factor = math.sqrt(2.0 / float(fan_in + fan_out))
            if self.bias is not None:
                nn.init.constant_(self.bias, 0)
                nn.init.constant_(self.bias_backward, 0)
        # Pytorch Default (Kaiming)
        elif self.init == 'kaiming':
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            # self.P.data = initialize_orthogonal(self.P)
            # nn.init.orthogonal_(self.P)
            # nn.init.orthogonal_(self.Q)
            
            nn.init.kaiming_uniform_(self.P, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Q, a=math.sqrt(5), mode='fan_in', nonlinearity='linear')

            
            # Scaling factor is the standard deviation of Kaiming init.
            if self.bias is not None:
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
                nn.init.uniform_(self.bias_backward, -bound, bound)
        # SVD-based initialization to approximate W with QP
        elif self.init == 'svd_approx':
            # Initialize weight normally first
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            
            # Use the separate function for SVD initialization
            self.init_svd_approx()
            
            # Initialize bias
            if self.bias is not None:
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
                nn.init.uniform_(self.bias_backward, -bound, bound)


    def forward(self, x: Tensor, gt=None) -> Tensor:
        return LinearGrad.apply(x, self.weight, self.P, self.Q, self.bias, self.bias_backward, gt)


    @staticmethod
    def gradient_clip(module, grad_input, grad_output):
        grad_input = list(grad_input)
        for i in range(len(grad_input)):
            if grad_input[i] is not None:
                grad_input[i] = torch.clamp(grad_input[i], -1, 1)
                #grad_input[i] = (grad_input[i] / torch.linalg.norm(grad_input[i]))
        return tuple(grad_input)
    
    @staticmethod
    def normalize_P(module, grad_input):
        module.P.data = F.normalize(module.P, p=2, dim=1)
        
        
        

def initialize_orthogonal(P):
    
    d, k = P.shape
    
    W = torch.randn(d, k)
    # Apply Gram-Schmidt orthogonalization
    for i in range(k):
        for j in range(i):
            # Subtract the projection of W[:, i] onto W[:, j]
            W[:, i] -= torch.dot(W[:, i], W[:, j]) * W[:, j]
        # Normalize W[:, i]
        W[:, i] = F.normalize(W[:, i], dim=0)
    return W