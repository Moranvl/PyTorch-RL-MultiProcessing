import torch
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from typing import Tuple
from pathlib import Path
from multiprocessing import Value
from .AgentBase import AgentBase
from .utils import build_mlp, optimizer_update, soft_update, layer_init_with_orthogonal
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

        self.state_dim = agent_args["state_dim"]
        self.action_dim = agent_args["action_dim"]
        self.horizon_len = agent_args["horizon_len"]
        self.repeat_times = agent_args["repeat_times"]
        self.batch_size = agent_args["batch_size"]
        self.soft_update_tau = agent_args["soft_update_tau"]
        self.clip_grad_norm = agent_args["clip_grad_norm"]
        self.gamma = agent_args["gamma"]

        # # Policy
        # self.cri.explore_rate = agent_args["explore_rate"]
        self.cri = QNet(
            agent_args["net_dims"], state_dim=self.state_dim, action_dim=self.action_dim,
            explore_rate=agent_args["explore_rate"],
        ).to(self.device)
        if role == "learner":
            self.cri_target = deepcopy(self.cri).to(self.device)
            # self.act = self.cri
            # self.act_optimizer = torch.optim.AdamW(self.act.parameters(), agent_args["learning_rate"])
            self.cri_optimizer = torch.optim.AdamW(self.cri.parameters(), agent_args["learning_rate"])
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
            writter.add_scalar(tag="exploit/sum_reward", scalar_value=sum_reward, global_step=self.exploit_step)
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
        # torch.save(self.act.state_dict(), model_path/f"act-{model_id}.pt")
        torch.save(self.cri.state_dict(), model_path / f"cri-{model_id}.pt")
        if not is_for_train:
            torch.save(self.cri_target.state_dict(), model_path / f"cri_target-{model_id}.pt")

    def load_model(self, model_id: int, model_path: Path, is_for_train: bool=True):
        model_path = model_path / f"{model_id}"
        # self.act.load_state_dict(torch.load(model_path/f"act-{model_id}.pt"))
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
        get_action = self.cri.get_action

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

class AgentDoubleDQN(AgentDQN):
    """
    Double Deep Q-Network algorithm. “Deep Reinforcement Learning with Double Q-learning”. 2015.
    """

    def __init__(self, env, share_model_id: Value, role: str, agent_args: dict):
        super().__init__(env, share_model_id, role, agent_args)

        self.cri = QNetTwin(
            agent_args["net_dims"], state_dim=self.state_dim, action_dim=self.action_dim,
            explore_rate=agent_args["explore_rate"],
        ).to(self.device)
        if role == "learner":
            self.cri_target = deepcopy(self.cri)
            self.cri_optimizer = torch.optim.AdamW(self.cri.parameters(), agent_args["learning_rate"])


    def get_obj_critic(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        # assert isinstance(update_t, int)
        with torch.no_grad():
            states, actions, rewards, undones, next_ss = buffer.sample(batch_size)  # next_ss: next states
            next_q = torch.min(*self.cri_target.get_q1_q2(next_ss)).max(dim=1, keepdim=True)[0]
            q_label = rewards + undones * self.gamma * next_q

        q_value1, q_value2 = [
            qs.squeeze(1).gather(dim=1, index=actions.long()) for qs in self.cri.get_q1_q2(states)
        ]
        obj_critic = td_error = self.criterion(q_value1, q_label) + self.criterion(q_value2, q_label)

        obj_actor = q_value1
        return obj_critic, obj_actor

class AgentD3QN(AgentDoubleDQN):  # Dueling Double Deep Q Network. (D3QN)
    def __init__(self, env, share_model_id: Value, role: str, agent_args: dict):
        super().__init__(env, share_model_id, role, agent_args)

        self.cri = QNetTwinDuel(
            agent_args["net_dims"], state_dim=self.state_dim, action_dim=self.action_dim,
            explore_rate=agent_args["explore_rate"],
        ).to(self.device)

        if role=="learner":
            self.cri_target = deepcopy(self.cri)
            self.cri_optimizer = torch.optim.AdamW(self.cri.parameters(), agent_args["learning_rate"])


'''Network'''

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

class QNetTwin(QNet):  # Double DQN
    def __init__(self, dims: [int], state_dim: int, action_dim: int, explore_rate=0.125):
        super().__init__(dims, state_dim, action_dim, explore_rate=0.125)
        self.net_state = build_mlp(dims=[state_dim, *dims])
        self.net_val1 = build_mlp(dims=[dims[-1], action_dim])  # Q value 1
        self.net_val2 = build_mlp(dims=[dims[-1], action_dim])  # Q value 2
        # self.soft_max = nn.Softmax(dim=-1)

        layer_init_with_orthogonal(self.net_val1[-1], std=0.1)
        layer_init_with_orthogonal(self.net_val2[-1], std=0.1)

    def get_q_value(self, state: Tensor) -> Tensor:
        s_enc = self.net_state(state)  # encoded state
        q_value = self.net_val1(s_enc)  # q value
        return q_value

    def get_q1_q2(self, state):
        s_enc = self.net_state(state)  # encoded state
        q_val1 = self.net_val1(s_enc)  # q value 1
        q_val2 = self.net_val2(s_enc)  # q value 2
        return q_val1, q_val2  # two groups of Q values

    def get_action(self, state):
        if self.explore_rate < torch.rand(1):
            action = self.get_q_value(state).argmax(dim=1, keepdim=True)
        else:
            action = torch.randint(self.action_dim, size=(state.shape[0], 1))
        return action

class QNetTwinDuel(QNetTwin):  # D3QN: Dueling Double DQN
    def __init__(self, dims: [int], state_dim: int, action_dim: int, explore_rate=0.125):
        super().__init__(dims, state_dim, action_dim, explore_rate=0.125)
        self.net_state = build_mlp(dims=[state_dim, *dims])
        self.net_adv1 = build_mlp(dims=[dims[-1], 1])  # advantage value 1
        self.net_val1 = build_mlp(dims=[dims[-1], action_dim])  # Q value 1
        self.net_adv2 = build_mlp(dims=[dims[-1], 1])  # advantage value 2
        self.net_val2 = build_mlp(dims=[dims[-1], action_dim])  # Q value 2

        layer_init_with_orthogonal(self.net_adv1[-1], std=0.1)
        layer_init_with_orthogonal(self.net_val1[-1], std=0.1)
        layer_init_with_orthogonal(self.net_adv2[-1], std=0.1)
        layer_init_with_orthogonal(self.net_val2[-1], std=0.1)

    def get_q_value(self, state):
        s_enc = self.net_state(state)  # encoded state
        q_val = self.net_val1(s_enc)  # q value
        q_adv = self.net_adv1(s_enc)  # advantage value
        q_value = q_val - q_val.mean(dim=1, keepdim=True) + q_adv  # one dueling Q value
        return q_value

    def get_q1_q2(self, state):
        s_enc = self.net_state(state)  # encoded state

        q_val1 = self.net_val1(s_enc)  # q value 1
        q_adv1 = self.net_adv1(s_enc)  # advantage value 1
        q_duel1 = q_val1 - q_val1.mean(dim=1, keepdim=True) + q_adv1

        q_val2 = self.net_val2(s_enc)  # q value 2
        q_adv2 = self.net_adv2(s_enc)  # advantage value 2
        q_duel2 = q_val2 - q_val2.mean(dim=1, keepdim=True) + q_adv2
        return q_duel1, q_duel2  # two dueling Q values