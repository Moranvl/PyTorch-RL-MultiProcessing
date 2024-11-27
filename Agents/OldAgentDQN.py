import torch
from torch import nn, Tensor
from copy import deepcopy
from typing import Tuple
from multiprocessing import Value
from .AgentBase import AgentBase
from .ReplayBuffer import ReplayBuffer
from .utils import build_mlp, soft_update, optimizer_update, copy_net_params


dqn_args = {
    "explore_rate": 0.25,
    "reward_scale": 1,
    "gamma": 0.99,
    "repeat_times": 1,
    "batch_size": 64,
    "soft_update_tau": 5e-3,  # 2 ** -8 ~= 5e-3. the tau of soft target update `net = (1-tau)*net + tau*net1`
    "clip_grad_norm": 3.0,  # 0.1 ~ 4.0, clip the gradient after normalization
    "learning_rate": 6e-5,  # the learning rate for network updating
}


class AgentDQN:
    """
    Deep Q-Network algorithm. “Human-Level Control Through Deep Reinforcement Learning”. Mnih V. et al.. 2015.
    net_dims: the middle layer dimension of MLP (MultiLayer Perceptron)
    state_dim: the dimension of state (the number of state vector)
    action_dim: the dimension of action (or the number of discrete action)
    gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
    args: the arguments for agent training. `args = Config()`
    """

    def __init__(self, args, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0):
        self.gamma = args["gamma"]
        self.reward_scale = args["reward_scale"]
        self.repeat_times = args["repeat_times"]
        self.batch_size = args["batch_size"]
        self.soft_update_tau = args["soft_update_tau"]
        self.clip_grad_norm = args["clip_grad_norm"]
        self.learning_rate = args["learning_rate"]

        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.cri = self.act = QNet(net_dims, state_dim, action_dim).to(self.device)
        self.act_target = self.cri_target = deepcopy(self.act)
        self.cri_optimizer = self.act_optimizer = torch.optim.AdamW(self.act.parameters(), self.learning_rate)
        self.act.explore_rate = args["explore_rate"]

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.criterion = torch.nn.SmoothL1Loss(reduction="mean")
        self.get_obj_critic = self.get_obj_critic_raw

    def explore_env(self, env, horizon_len: int) -> Tuple[Tensor, ...]:
        """
        Collect trajectories through the actor-environment interaction for a **single** environment instance.
        """
        states = torch.zeros((horizon_len + 1, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, 1), dtype=torch.int32).to(self.device)  # different
        rewards = torch.zeros((horizon_len, 1), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, 1), dtype=torch.bool).to(self.device)

        state, _ = env.reset()
        state = states[0] = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        get_action = self.act.get_action

        for t in range(horizon_len):
            with torch.no_grad():
                action = get_action(state)  # different

            ary_action = action.detach().cpu().numpy().item()
            ary_state, reward, terminated, truncated, _ = env.step(ary_action)  # next_state
            done = terminated or truncated
            state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            states[t + 1] = state
            actions[t] = action
            rewards[t] = reward
            dones[t] = done

            if done:
                break

        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)
        return states, actions, rewards, undones

    def update_net(self, buffer: ReplayBuffer) -> Tuple[float, ...]:
        """update network"""
        obj_critics = 0.0
        obj_actors = 0.0

        update_times = int(buffer.add_size * self.repeat_times)
        assert update_times >= 1
        for _ in range(update_times):
            obj_critic, q_value = self.get_obj_critic(buffer, self.batch_size)
            obj_critics += obj_critic.item()
            obj_actors += q_value.mean().item()
            optimizer_update(self.cri_optimizer, obj_critic, self.clip_grad_norm)
            soft_update(self.cri_target, self.cri, self.soft_update_tau)
        return obj_critics / update_times, obj_actors / update_times

    def get_obj_critic_raw(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        """
        Calculate the loss of the network and predict Q values with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            states, actions, rewards, undones, next_ss = buffer.sample(batch_size)  # next_ss: next states
            next_qs = self.cri_target(next_ss).max(dim=1, keepdim=True)[0]  # next q_values
            q_labels = rewards + undones * self.gamma * next_qs

        q_values = self.cri(states).gather(1, actions.long())
        obj_critic = self.criterion(q_values, q_labels)
        return obj_critic, q_values

    def get_params(self):
        params = deepcopy(self.act)
        params.cpu()
        return params

    def update_params(self, parameters):
        self.act = parameters
        self.act.cpu()
        # print(self.act(torch.zeros(self.state_dim)))


class QNet(nn.Module):
    def __init__(self, dims: [int], state_dim: int, action_dim: int, explore_rate=0.125):
        super().__init__()
        self.explore_rate = explore_rate
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = build_mlp(dims=[state_dim, *dims, action_dim])

    def forward(self, state):
        value = self.net(state)
        return value  # Q values for multiple actions

    def get_action(self, state):
        if self.explore_rate < torch.rand(1):
            action = self.net(state).argmax(dim=1, keepdim=True)
        else:
            action = torch.randint(self.action_dim, size=(state.shape[0], 1))
        return action
