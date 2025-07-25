import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math

class LinearGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, P, Q, bias, S, beta):
        ctx.save_for_backward(x, weight, P, Q, bias, S)
        ctx.beta = beta
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, P_old, Q, bias, S_old = ctx.saved_tensors
        beta = ctx.beta
        
        # --- Vectorized PAST Approximation (No Loops) ---
        with torch.no_grad():
            E = grad_output.reshape(-1, grad_output.shape[-1])
            Y = E @ P_old
            H = Y @ S_old
            gamma_inv = beta + (Y * H).sum(dim=1, keepdim=True)
            # [B, r] / [B, 1] -> [B, r]
            L = H / gamma_inv

            # 3. Calculate the update for P and S using matrix multiplication
            # This replaces the sum of outer products (g @ l.T)
            # [d, B] @ [B, r] -> [d, r]
            delta_P = (E - Y @ P_old.T).T @ L
            P_new = P_old + delta_P
            
            # This replaces the sum of outer products (l @ h.T)
            # [r, B] @ [B, r] -> [r, r]
            delta_S = L.T @ H
            S_new = (1.0 / beta) * (S_old - delta_S)

        # --- "Fake Gradient" Trick ---
        P_old.data = P_new
        S_old.data = S_new

        # --- Standard Gradient Calculations (Unchanged) ---
        grad_x = grad_weight = grad_Q = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output @ (P_old @ Q)
        if ctx.needs_input_grad[1]:
            x_reshaped = x.reshape(-1, x.shape[-1])
            grad_output_reshaped = grad_output.reshape(-1, grad_output.shape[-1])
            grad_weight = grad_output_reshaped.T @ x_reshaped
        if ctx.needs_input_grad[3]:
            grad_q_signal = grad_output @ P_old
            grad_q_signal_reshaped = grad_q_signal.reshape(-1, grad_q_signal.shape[-1])
            grad_Q = grad_q_signal_reshaped.T @ x.reshape(-1, x.shape[-1])
        if bias is not None and ctx.needs_input_grad[4]:
            grad_bias = grad_output.sum(dim=0)
            
        return grad_x, grad_weight, None, grad_Q, grad_bias, None, None

# ==============================================================================
# The Linear class and optimizer setup remain IDENTICAL to the previous version.
# Only the `backward` method above needs to be replaced.
# ==============================================================================
class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True, past_beta: float = 0.99, update_Q: bool = False) -> None:
        super(Linear, self).__init__(in_features, out_features, bias)
        self.rank = rank
        self.past_beta = past_beta
        self.P = nn.Parameter(torch.Tensor(out_features, self.rank))
        self.Q = nn.Parameter(torch.Tensor(self.rank, in_features), requires_grad=update_Q)
        self.S = nn.Parameter(torch.Tensor(self.rank, self.rank))
        self.init_parameters()
        
    def init_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.orthogonal_(self.P)
        nn.init.kaiming_uniform_(self.Q, a=math.sqrt(5), mode='fan_in', nonlinearity='linear')
        with torch.no_grad():
            self.S.copy_((1.0 / 1e-6) * torch.eye(self.rank, device=self.S.device))
                    
    def forward(self, x: Tensor) -> Tensor:
        return LinearGrad.apply(x, self.weight, self.P, self.Q, self.bias, self.S, self.past_beta)