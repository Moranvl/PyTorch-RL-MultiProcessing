import torch
from torch import nn, Tensor
from torch.nn.utils import clip_grad_norm_


def build_mlp(dims: [int], activation: nn = None, if_raw_out: bool = True) -> nn.Sequential:
    """
    build MLP (MultiLayer Perceptron)

    dims: the middle dimension, `dims[-1]` is the output dimension of this network
    activation: the activation function
    if_remove_out_layer: if remove the activation function of the output layer.
    """
    if activation is None:
        activation = nn.ReLU
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), activation()])
    if if_raw_out:
        del net_list[-1]  # delete the activation function of the output layer to keep raw output
    return nn.Sequential(*net_list)


def soft_update(target_net: torch.nn.Module, current_net: torch.nn.Module, tau: float):
    """soft update target network via current network

    target_net: update target network via current network to make training more stable.
    current_net: current network update via an optimizer
    tau: tau of soft target update: `target_net = target_net * (1-tau) + current_net * tau`
    """
    for tar, cur in zip(target_net.parameters(), current_net.parameters()):
        tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))


def optimizer_update(optimizer: torch.optim, objective: Tensor, clip_grad_norm):
    """minimize the optimization objective via update the network parameters

    optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
    objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
    """
    optimizer.zero_grad()
    objective.backward()
    clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=clip_grad_norm)
    optimizer.step()
