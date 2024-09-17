import math

import torch
import torch.nn as nn
from torch import Tensor
from torch import autograd
from typing import Union
from torch.nn.common_types import _size_2_t


#from biotorch.layers.metrics import compute_matrix_angle

class Conv2dGrad(autograd.Function):
    """
    Autograd Function that Does a backward pass using the weight_backward matrix of the layer
    """
    @staticmethod
    def forward(context, input, weight, P, Q, bias, bias_backward, rank, stride, padding, dilation, groups):
        context.stride, context.padding, context.dilation, context.groups, context.rank = stride, padding, dilation, groups, rank
        context.save_for_backward(input, weight, P,Q, bias, bias_backward)
        output = torch.nn.functional.conv2d(input,
                                            weight,
                                            bias=bias,
                                            stride=stride,
                                            padding=padding,
                                            dilation=dilation,
                                            groups=groups)
        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, P, Q, bias, bias_backward = context.saved_tensors
        grad_input = grad_weight = grad_P = grad_Q = grad_bias = grad_bias_backward = None
        
        grad_input_P = None
        # Gradient input
        if context.needs_input_grad[0]:
            # Use the FA constant weight matrix to compute the gradient
            
            b,c,h,w = input.shape
            input_size = torch.Size((b, context.rank, h, w))
            
            
            
            intermediate_grad = torch.nn.grad.conv2d_input(input_size=input_size,
                                                    weight=P,
                                                    grad_output=grad_output,
                                                    stride=context.stride,
                                                    padding=0,
                                                    dilation=context.dilation,
                                                    groups=context.groups)
            
            grad_input = torch.nn.grad.conv2d_input(input_size=input.shape,
                                        weight=Q,
                                        grad_output=intermediate_grad,
                                        stride=context.stride,
                                        padding=context.padding,
                                        dilation=context.dilation,
                                        groups=context.groups)
            
            
            grad_out_transpose = grad_output.permute(0, 2, 3, 1)
            grad_out_transpose = grad_out_transpose.reshape(-1, grad_out_transpose.shape[-1])
            grad_out_transpose = (grad_out_transpose - grad_out_transpose.mean(0)) / (grad_out_transpose.std(0) + 1e-6)
            
            P_s = P.squeeze(-1, -2)
            grad_P = -1 * (torch.eye(P_s.shape[0]).to(P_s.device) - P_s@P_s.T).mm(grad_out_transpose.T.mm(grad_out_transpose).mm(P_s))[..., None, None]
            
            # P_s = P_s / torch.linalg.norm(P_s, dim = 1)[...,None]+ 1e-8
            # P = P_s[..., None, None]
                        
            grad_Q = torch.nn.grad.conv2d_weight(input=input,
                    weight_size=Q.shape,
                    grad_output=intermediate_grad,
                    stride=context.stride,
                    padding=context.padding,
                    dilation=context.dilation,
                    groups=context.groups)
            
            

        # Gradient weights
        if context.needs_input_grad[1]:
            
            grad_weight = torch.nn.grad.conv2d_weight(input=input,
                                                      weight_size=weight.shape,
                                                      grad_output=grad_output,
                                                      stride=context.stride,
                                                      padding=context.padding,
                                                      dilation=context.dilation,
                                                      groups=context.groups)

        # Gradient bias
        if bias is not None and context.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).sum(2).sum(1)            


        # Return the same number of parameters
        return grad_input, grad_weight, grad_P, grad_Q, grad_bias, grad_bias_backward, None, None, None, None, None

class Conv2d(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            rank: int,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            layer_config: dict = None
    ):

        super(Conv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode
        )

        self.layer_config = layer_config

        if self.layer_config is None:
            self.layer_config = dict()
            
        if "options" not in self.layer_config:
            self.layer_config["options"] = {
                "constrain_weights": False,
                "gradient_clip": False,
                "init": "kaiming"
            }
            
        #TODO ccheck whats the right way to construct the backward kernels regarding, bias, and kernel size
        self.rank = rank
        self.options = self.layer_config["options"]
        self.init = self.options["init"]
        
        self.P = nn.Parameter(torch.Tensor(out_channels, rank, 1, 1), requires_grad=True)
        self.Q = nn.Parameter(torch.Tensor(rank, in_channels, kernel_size, kernel_size), requires_grad=True)
        
        if self.bias is not None:
            self.bias_backward = nn.Parameter(torch.Tensor(self.bias.size()), requires_grad=False)
        else:
            self.register_parameter("bias", None)
            self.bias_backward = None

        self.init_parameters()


        if "constrain_weights" in self.options and self.options["constrain_weights"]:
            self.norm_initial_weights = torch.linalg.norm(self.weight)
        
        
        # self.register_full_backward_hook(self.gradient_clip)
        #if "gradient_clip" in self.options and self.options["gradient_clip"]:
            # self.register_full_backward_hook(self.gradient_clip)

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
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.P, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Q, a=math.sqrt(5), mode='fan_in', nonlinearity='linear')
            # Scaling factor is the standard deviation of Kaiming init.
            self.scaling_factor = 1 / math.sqrt(3 * fan_in)
            if self.bias is not None:
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
                nn.init.uniform_(self.bias_backward, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            # Based on "Feedback alignment in deep convolutional networks" (https://arxiv.org/pdf/1812.06488.pdf)
            # Constrain weight magnitude
            if "constrain_weights" in self.options and self.options["constrain_weights"]:
                self.weight = torch.nn.Parameter(self.weight * self.norm_initial_weights /
                                                 torch.linalg.norm(self.weight))
                
        return Conv2dGrad.apply(x,
                                self.weight,
                                self.P,
                                self.Q,
                                self.bias,
                                self.bias_backward,
                                self.rank,
                                self.stride,
                                self.padding,
                                self.dilation,
                                self.groups)


    @staticmethod
    def gradient_clip(module, grad_input, grad_output):
        grad_input = list(grad_input)
        for i in range(len(grad_input)):
            if grad_input[i] is not None:
                #grad_input[i] = torch.clamp(grad_input[i], -1, 1)
                grad_input[i] = (grad_input[i] / torch.linalg.norm(grad_input[i]))
        return tuple(grad_input)
