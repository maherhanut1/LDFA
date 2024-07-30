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
    def forward(context, input, weight, B, bias=None, bias_backward=None):
        context.save_for_backward(input, weight, B, bias, bias_backward)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, B, bias, bias_backward = context.saved_tensors
        grad_input = grad_weight = grad_B = grad_bias = grad_bias_backward = None
        # Gradient input
        if context.needs_input_grad[0]:
            # Use the B matrix to compute the gradient
            grad_input = grad_output.mm(B)
            
        # Gradient weights
        if context.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        
        if context.needs_input_grad[2]:
            grad_B = grad_output.t().mm(input)
        # Gradient bias
        if bias is not None and context.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_B, grad_bias, grad_bias_backward


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, layer_config: dict = None) -> None:
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
        
        self.B = nn.Parameter(torch.Tensor(self.weight.size()), requires_grad=True)
        
        if self.bias is not None:
            self.bias_backward = nn.Parameter(torch.Tensor(self.bias.size()), requires_grad=False)
        else:
            self.register_parameter("bias", None)
            self.bias_backward = None

        self.init_parameters()

        if "gradient_clip" in self.options and self.options["gradient_clip"]:
            self.register_backward_hook(self.gradient_clip)

    def init_parameters(self) -> None:
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
        # Xavier initialization
        if self.init == "xavier":
            nn.init.xavier_uniform_(self.weight)
            nn.init.xavier_uniform_(self.B)
            
            # Scaling factor is the standard deviation of xavier init.
            self.scaling_factor = math.sqrt(2.0 / float(fan_in + fan_out))
            if self.bias is not None:
                nn.init.constant_(self.bias, 0)
                nn.init.constant_(self.bias_backward, 0)
        # Pytorch Default (Kaiming)
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
            # Scaling factor is the standard deviation of Kaiming init.
            self.scaling_factor = 1 / math.sqrt(3 * fan_in)
            if self.bias is not None:
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
                nn.init.uniform_(self.bias_backward, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return LinearGrad.apply(x, self.weight, self.B, self.bias, self.bias_backward)


    @staticmethod
    def gradient_clip(module, grad_input, grad_output):
        grad_input = list(grad_input)
        for i in range(len(grad_input)):
            if grad_input[i] is not None:
                grad_input[i] = torch.clamp(grad_input[i], -1, 1)
        return tuple(grad_input)