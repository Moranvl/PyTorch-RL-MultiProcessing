import torch
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from typing import Tuple
from pathlib import Path
from multiprocessing import Value
from .AgentBase import AgentBase
from .utils import build_mlp, optimizer_update, soft_update
from .ReplayBuffer import ReplayBuffer

args = {
    "state_dim": 8,
    "action_dim": 8,
    "horizon_len": 200,

    "net_dims": [256, 256, 256],

    "explore_rate": 0.25,
    "gamma": 0.99,
    "repeat_times": 1,
    "batch_size": 128,
    "soft_update_tau": 5e-3,  # 2 ** -8 ~= 5e-3. the tau of soft target update `net = (1-tau)*net + tau*net1`
    "clip_grad_norm": 3.0,  # 0.1 ~ 4.0, clip the gradient after normalization
    "learning_rate": 6e-5,  # the learning rate for network updating
    "buffer":{
        "max_size": int(1e6),
        "action_dim": 1,
    }
}

class AgentDQN(AgentBase):
    """
    Deep Q-Network algorithm.
    """
    def __init__(self, env, share_model_id: Value, role: str, agent_args: dict):
        super().__init__(env, share_model_id, role, agent_args)
        # Args
        self.exploit_step = 0
        self.device = None
        if role == 'learner' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif role == 'worker' or role =='learner':
            self.device = torch.device('cpu')
        else:
            raise ValueError(f"role is error! role: {role}")
        self.state_dim = agent_args["state_dim"]
        self.action_dim = agent_args["action_dim"]
        self.horizon_len = agent_args["horizon_len"]
        self.repeat_times = agent_args["repeat_times"]
        self.batch_size = agent_args["batch_size"]
        self.soft_update_tau = agent_args["soft_update_tau"]
        self.clip_grad_norm = agent_args["clip_grad_norm"]
        self.gamma = agent_args["gamma"]

        # # Policy
        self.act = QNet(
            agent_args["net_dims"], state_dim=self.state_dim, action_dim=self.action_dim
        ).to(self.device)
        self.cri = self.act
        self.cri_target = deepcopy(self.cri).to(self.device)
        self.act_optimizer = torch.optim.AdamW(self.act.parameters(), agent_args["learning_rate"])
        self.cri_optimizer = torch.optim.AdamW(self.cri.parameters(), agent_args["learning_rate"])
        self.act.explore_rate = agent_args["explore_rate"]
        self.criterion = torch.nn.SmoothL1Loss(reduction="mean")

        # ReplayBuffer
        self.replay_buffer = ReplayBuffer(
            max_size=agent_args["buffer"]["max_size"],
            state_dim=agent_args["state_dim"],
            action_dim=agent_args["buffer"]["action_dim"],
            batch_size=agent_args["batch_size"],
            gpu_id=0 if role == 'learner' else -1,
        )

    def explore(self, model_id: int, model_path: Path):
        if model_id != 0:
            self.load_model(model_id, model_path)
        data = self.explore_env(self.env, self.horizon_len)
        return data

    def exploit(self, writter: SummaryWriter or None):
        states, actions, rewards, undones = self.explore_env(self.env, self.horizon_len)
        sum_reward = torch.sum(rewards)
        if writter is not None:
            writter.add_scalar(tag="train/sum_reward", scalar_value=sum_reward, global_step=self.exploit_step)
            self.exploit_step += 1
        return sum_reward

    def update_buffer(self, data) -> None:
        self.replay_buffer.update(data)

    def learn(self, writter: SummaryWriter or None):
        obj_cri, obj_act = self.update_net(self.replay_buffer)
        if writter is not None:
            writter.add_scalar(tag="loss/cri", scalar_value=obj_cri, global_step=self.exploit_step)
            writter.add_scalar(tag="loss/act", scalar_value=obj_act, global_step=self.exploit_step)

    def save_model(self, model_id: int, model_path: Path, is_for_train: bool=True):
        model_path = model_path / f"{model_id}"
        model_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.act.state_dict(), model_path/f"act-{model_id}.pt")
        torch.save(self.cri.state_dict(), model_path / f"cri-{model_id}.pt")
        if not is_for_train:
            torch.save(self.cri_target.state_dict(), model_path / f"cri_target-{model_id}.pt")

    def load_model(self, model_id: int, model_path: Path, is_for_train: bool=True):
        model_path = model_path / f"{model_id}"
        self.act.load_state_dict(torch.load(model_path/f"act-{model_id}.pt"))
        self.cri.load_state_dict(torch.load(model_path / f"cri-{model_id}.pt"))
        if not is_for_train:
            self.cri_target.load_state_dict(torch.load(model_path / f"cri_target-{model_id}.pt"))

    def close(self):
        pass

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

    def get_obj_critic(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
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