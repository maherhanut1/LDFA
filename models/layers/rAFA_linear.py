import math
import torch
import torch.nn as nn
from torch import Tensor

from torch import autograd


class LinearGrad(autograd.Function):
    """
    Autograd Function that Does a backward pass using the B matrix of the layer
    """
    @staticmethod
    # Same as reference linear function, but with additional weight tensor for backward
    def forward(context, input, weight, P, Q, bias=None, bias_backward=None, gt=None):
        
        output = input.mm(weight.t())
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
            grad_input_intermediate = grad_output.mm(P)
            grad_input = grad_input_intermediate.mm(Q)
            
        # Gradient weights
        if context.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        
        if context.needs_input_grad[2]:
            if gt is not None:
                one_hot_targets = torch.zeros(grad_output.shape).to(gt.device)
                one_hot_targets.scatter_(torch.tensor(1).to(gt.device), gt.unsqueeze(1), 1.)
                grad_P = -1 * (torch.eye(P.shape[0]).to(P.device) - P@P.T).mm(one_hot_targets.T.mm(one_hot_targets).mm(P)) / grad_output.shape[0]
            else:
                grad_P = -1 * (torch.eye(P.shape[0]).to(P.device) - P@P.T).mm(grad_output.T.mm(grad_output).mm(P)) / grad_output.shape[0]
                
           # grad_P = grad_P.t()  #/ torch.linalg.norm(grad_P)
            P = P / torch.linalg.norm(P, dim = 1)[...,None]+ 1e-8
            
        if grad_input_intermediate is not None and context.needs_input_grad[3]:
            grad_Q = grad_input_intermediate.t().mm(input) #/ (grad_output.shape[0])
            
        # Gradient bias
        if bias is not None and context.needs_input_grad[4]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_P, grad_Q, grad_bias, grad_bias_backward, None


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True, layer_config: dict = None, update_P = True, update_Q = False, requires_gt = False) -> None:
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
        self.register_full_backward_hook(self.gradient_clip)
        
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
            nn.init.kaiming_uniform_(self.P, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Q, a=math.sqrt(5))
            # Scaling factor is the standard deviation of Kaiming init.
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