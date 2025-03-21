import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from models.layers.GLobal_e import Ewrapper

class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, rank: int, loss_module, bias: bool = True, layer_config: dict = None, update_P = True, update_Q = False, requires_gt = False) -> None:
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
        self.P = nn.Parameter(torch.Tensor(512, self.rank), requires_grad=update_P)
        
        self._loss_module = loss_module
        
        if self.bias is not None:
            self.bias_backward = nn.Parameter(torch.Tensor(self.bias.size()), requires_grad=False)
        else:
            self.register_parameter("bias", None)
            self.bias_backward = None

        self.init_parameters()
        # self.register_forward_pre_hook(self.normalize_P)
        self.register_full_backward_hook(self.DFA_backward_hook)
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
        elif self.init == 'kaiming':
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.P, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Q, a=math.sqrt(5), mode='fan_in', nonlinearity='linear')
            # Scaling factor is the standard deviation of Kaiming init.
            if self.bias is not None:
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
                nn.init.uniform_(self.bias_backward, -bound, bound)
            
                    
            
    def forward(self, x: Tensor, gt=None) -> Tensor:
        self.inputs = x.clone()
        self.gt = gt.clone() if gt is not None else None
        output = x.mm(self.weight.t())
        if self.bias is not None:
            output += self.bias.unsqueeze(0).expand_as(output)
        return output


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
        
    
    @staticmethod
    def DFA_backward_hook(module, grad_input, grad_output):
        
        if grad_input[0] is None:
            return grad_input
        else:
            dfa_grad_output = Ewrapper.get_E()[0]
            module_inputs = module.inputs
           
            grad_input_intermediate = dfa_grad_output.mm(module.P)
            grad_dfa = grad_input_intermediate.mm(module.Q)
            
            if module.P.requires_grad:
#                one_hot_targets = dfa_grad_output
                # gt = module.gt
                # one_hot_targets = torch.zeros(dfa_grad_output.shape).to(gt.device)
                # one_hot_targets.scatter_(torch.tensor(1).to(gt.device), gt.unsqueeze(1), 1.)
                one_hot_targets = dfa_grad_output
                one_hot_targets_mean = one_hot_targets.mean(0)
                one_hot_targets_std = one_hot_targets.std(0)
                normalized_one_hot_targets = (one_hot_targets - one_hot_targets_mean) / (one_hot_targets_std + 1e-8)
                grad_P = -1 * (torch.eye(module.P.shape[0]).to(module.P.device) - module.P@module.P.T).mm(normalized_one_hot_targets.T.mm(normalized_one_hot_targets).mm(module.P))
                module.P.grad = grad_P
            
            if module.Q.requires_grad:
                grad_Q = grad_input_intermediate.t().mm(module_inputs)
                module.Q.grad = grad_Q
            
            if len(grad_input) == 2:
                return grad_dfa, grad_input[1]
            else:
                return (grad_dfa,) +  grad_input[1:]