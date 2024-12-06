import torch
import numpy as np
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from typing import List, Tuple
from copy import deepcopy
from pathlib import Path
from multiprocessing import Value
from .AgentBase import AgentBase
from .ReplayBuffer import ReplayBuffer
from .utils import build_mlp, layer_init_with_orthogonal, optimizer_update, soft_update

args = {
    # envs
    "action_space": {
        "high": [1,],
        "low": [-1,],
    },

    "state_dim": 8,
    "action_dim": 8,
    "horizon_len": 200,
    "repeat_times": 25,

    "net_dims": {
        "act": [256, 256, 256],
        "cri": [256, 256, 256],
    },


    "learning_rate": {
        "act": 5e-6,
        "cri": 1e-5,
        "alpha": 1e-6,
    },  # the learning rate for network updating
    "update_tau": 0.05,

    "explore_rate": 0.25,
    "gamma": 0.99,
    "batch_size": 128,
    "soft_update_tau": 5e-3,  # 2 ** -8 ~= 5e-3. the tau of soft target update `net = (1-tau)*net + tau*net1`
    "clip_grad_norm": 3.0,  # 0.1 ~ 4.0, clip the gradient after normalization
    "buffer":{
        "max_size": int(1e6),
    }
}

class AgentSAC(AgentBase):
    """
    Soft Actor Critic in Continuous Action Space.
    """
    def __init__(self, env, share_model_id: Value, role: str, agent_args: dict):
        super().__init__(env, share_model_id, role, agent_args)
        self.exploit_step = 0
        self.state_dim = agent_args["state_dim"]
        self.action_dim = agent_args["action_dim"]
        self.horizon_len = agent_args["horizon_len"]
        self.repeat_times = agent_args["repeat_times"]
        self.batch_size = agent_args["batch_size"]
        self.gamma = agent_args["gamma"]
        self.clip_grad_norm = agent_args["clip_grad_norm"]

        # Policy
        limit = {"high": agent_args["action_space"]["high"], "low": agent_args["action_space"]["low"]}
        self.act = ActorSAC(
            agent_args["net_dims"]["act"], agent_args["state_dim"], agent_args["action_dim"],
            limit=limit, device=self.device
        ).to(self.device)
        if role == "learner":
            self.cri = CriticEnsemble(
                agent_args["net_dims"]["cri"], agent_args["state_dim"], agent_args["action_dim"]
            ).to(self.device)
            self.cri_target = deepcopy(self.cri)
            self.act_optimizer = torch.optim.AdamW(self.act.parameters(), lr=agent_args["learning_rate"]["act"])
            self.cri_optimizer = torch.optim.AdamW(self.cri.parameters(), lr=agent_args["learning_rate"]["cri"])
            self.criterion = torch.nn.SmoothL1Loss(reduction="mean")

        # Special for SAC
        self.alpha_log = torch.tensor((-1,), dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.AdamW((self.alpha_log,), lr=agent_args["learning_rate"]["alpha"])
        self.target_entropy = np.log(agent_args["action_dim"])
        self.soft_update_tau = agent_args["update_tau"]

        # Reply Buffer
        self.replay_buffer = ReplayBuffer(
            max_size=agent_args["buffer"]["max_size"],
            state_dim=agent_args["state_dim"],
            action_dim=agent_args["action_dim"],
            batch_size=agent_args["batch_size"],
            gpu_id=0 if role == 'learner' else -1,
        )

    def explore(self, model_id: int, model_path: Path):
        """
        explore all the env
        :return:  <s, a, r, s'> (to self.update_buffer)
        """
        if model_id != 0:
            self.load_model(model_id, model_path, is_for_train=True)
        data = self.explore_env(self.env, self.horizon_len)
        return data

    def exploit(self, writter: SummaryWriter or None):
        """
        exploit the env
        :type writter: tensorboard
        :return:  exploit kpi
        """
        states, actions, rewards, undones = self.explore_env(self.env, self.horizon_len)
        sum_reward = torch.sum(rewards)
        if writter is not None:
            writter.add_scalar(tag="train/sum_reward", scalar_value=sum_reward, global_step=self.exploit_step)
            self.exploit_step += 1
        return sum_reward

    def update_buffer(self, data) -> None:
        """
        update the new data to buffer
        :param data: new data from the self.explore
        :return:  None
        """
        self.replay_buffer.update(data)

    def learn(self, writter: SummaryWriter or None):
        """
        Update the policy of the agent
        :return: None
        """
        buffer = self.replay_buffer
        objs_critic = []
        objs_actor = []
        objs_alpha = []

        # if self.lambda_fit_cum_r != 0:
        #     buffer.update_cum_rewards(get_cumulative_rewards=self.get_cumulative_rewards)

        torch.set_grad_enabled(True)
        # update_times = int(buffer.cur_size * self.repeat_times / self.batch_size)
        update_times = int(self.repeat_times * (1 + buffer.cur_size / buffer.max_size))
        for update_t in range(update_times):
            assert isinstance(update_t, int)
            with torch.no_grad():
                state, action, reward, undone, next_state = buffer.sample(self.batch_size)

                next_action, next_logprob = self.act.get_action_logprob(next_state)  # stochastic policy
                next_q = torch.min(self.cri_target.get_q_values(next_state, next_action), dim=1, keepdim=True)[0]
                alpha = self.alpha_log.exp()
                q_label = reward + undone * self.gamma * (next_q - next_logprob * alpha)

            '''objective of critic (loss function of critic)'''
            q_values = self.cri.get_q_values(state, action)
            q_labels = q_label.view((-1, 1)).repeat(1, q_values.shape[1])
            obj_critic = self.criterion(q_values, q_labels)

            optimizer_update(self.cri_optimizer, obj_critic, self.clip_grad_norm)
            soft_update(self.cri_target, self.cri, self.soft_update_tau)

            '''objective of alpha (temperature parameter automatic adjustment)'''
            action_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
            obj_alpha = (self.alpha_log * (self.target_entropy - logprob).detach()).mean()
            optimizer_update(self.alpha_optim, obj_alpha, self.clip_grad_norm)

            '''objective of actor'''
            alpha = self.alpha_log.exp().detach()
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2)

            q_value_pg = self.cri_target(state, action_pg).mean()
            obj_actor = -(q_value_pg - logprob * alpha).mean()
            optimizer_update(self.act_optimizer, obj_actor, self.clip_grad_norm)

            '''remember'''
            objs_critic.append(obj_critic.item())
            objs_actor.append(obj_actor.item())
            objs_alpha.append(obj_alpha.item())
        torch.set_grad_enabled(False)

        obj_avg_critic = np.nanmean(objs_critic) if len(objs_critic) else 0.0
        obj_avg_actor = np.nanmean(objs_actor) if len(objs_actor) else 0.0
        obj_avg_alpha = np.nanmean(objs_alpha) if len(objs_alpha) else 0.0


        if writter is not None:
            writter.add_scalar(tag="loss/cri", scalar_value=obj_avg_critic, global_step=self.exploit_step)
            writter.add_scalar(tag="loss/act", scalar_value=obj_avg_actor, global_step=self.exploit_step)
            writter.add_scalar(tag="loss/alpha", scalar_value=obj_avg_alpha, global_step=self.exploit_step)

    def save_model(self, model_id: int, model_path: Path, is_for_train: bool):
        """
        Save the model of the agent SAC
        :type is_for_train: for training
        :param model_path: the path to the model
        :param model_id: the folder of model to save
        :return: None
        """
        model_path = model_path / f"{model_id}"
        model_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.act.state_dict(), model_path / f"act-{model_id}.pt")
        if not is_for_train:
            torch.save(self.cri.state_dict(), model_path / f"cri-{model_id}.pt")
            torch.save(self.cri_target.state_dict(), model_path / f"cri_target-{model_id}.pt")

    def load_model(self, model_id: int, model_path: Path, is_for_train: bool):
        """
        Save the model of the agent SAC
        :param is_for_train:  for training
        :param model_path: the path to the model
        :param model_id: the folder of model to load
        :return: None
        """
        model_path = model_path / f"{model_id}"
        self.act.load_state_dict(torch.load(model_path / f"act-{model_id}.pt"))
        if not is_for_train:
            self.cri.load_state_dict(torch.load(model_path / f"cri-{model_id}.pt"))
            self.cri_target.load_state_dict(torch.load(model_path / f"cri_target-{model_id}.pt"))

    def close(self):
        """
        close all cuda tensors.
        :return: None
        """
        pass

    def explore_env(self, env, horizon_len: int) -> Tuple[Tensor, ...]:
        """
        Collect trajectories through the actor-environment interaction for a **single** environment instance.
        """
        states = torch.zeros((horizon_len + 1, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, 1), dtype=torch.float32).to(self.device)  # different
        rewards = torch.zeros((horizon_len, 1), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, 1), dtype=torch.bool).to(self.device)

        state, _ = env.reset()
        state = states[0] = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        get_action = self.act.get_action

        for t in range(horizon_len):
            with torch.no_grad():
                action, _ = get_action(state)

            # ary_action = action.detach().cpu().numpy().item()
            ary_action = action.detach().cpu().numpy()[0]
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


class ActorSAC(torch.nn.Module):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, limit: dict, device):
        super().__init__()
        self.normal = torch.distributions.normal.Normal(
            loc=torch.tensor(0.0).to(device), scale=torch.tensor(1.0).to(device)
        )
        self.action_dim = action_dim
        limit_high = torch.tensor(limit["high"])
        limit_low = torch.tensor(limit["low"])
        self.center = ((limit_high + limit_low) / 2).to(device)
        self.scale = ((limit_high - limit_low) / 2).to(device)

        self.net_s = build_mlp(dims=[state_dim, *net_dims], if_raw_out=False)  # network of encoded state
        self.net_a = build_mlp(dims=[net_dims[-1], action_dim * 2])  # the average and log_std of action
        layer_init_with_orthogonal(self.net_a[-1], std=0.1)

    def forward(self, state):
        s_enc = self.net_s(state)  # encoded state
        a_avg = self.net_a(s_enc)[:, :self.action_dim]
        return self.process_action(a_avg.tanh())  # action

    def resample(self, avg: Tensor, std: Tensor):
        shape = avg.shape
        eps = self.normal.sample(shape)
        log_prob = self.normal.log_prob(eps)
        return avg + eps * std, log_prob

    def get_action(self, state):
        s_enc = self.net_s(state)                               # encoded state
        a_avg, a_std_log = self.net_a(s_enc).chunk(2, dim=1)    # chunk the data into 2 blocks
        a_std = a_std_log.clamp(-16, 2).exp()

        action, log_prob = self.resample(a_avg, a_std)
        # action (re-parameterize)
        return self.process_action(action.tanh()), log_prob

    def get_action_logprob(self, state):
        s_enc = self.net_s(state)  # encoded state
        a_avg, a_std_log = self.net_a(s_enc).chunk(2, dim=1)
        a_std = a_std_log.clamp(-16, 2).exp()

        action, log_prob = self.resample(a_avg, a_std)
        action_tanh = action.tanh()

        log_prob -= (-action_tanh.pow(2) + 1.000001).log()  # fix logprob using the derivative of action.tanh()
        # return self.process_action(action_tanh), log_prob.sum(1)
        return self.process_action(action_tanh), log_prob

    def process_action(self, action):
        return self.scale * action + self.center

class CriticEnsemble(torch.nn.Module):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, num_ensembles: int = 4):
        super().__init__()
        self.encoder_sa = build_mlp(dims=[state_dim + action_dim, net_dims[0]])  # encoder of state and action
        self.decoder_qs = []
        for net_i in range(num_ensembles):
            decoder_q = build_mlp(dims=[*net_dims, 1])
            layer_init_with_orthogonal(decoder_q[-1], std=0.5)

            self.decoder_qs.append(decoder_q)
            setattr(self, f"decoder_q{net_i:02}", decoder_q)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        values = self.get_q_values(state=state, action=action)
        value = values.mean(dim=-1, keepdim=True)
        return value  # Q value

    def get_q_values(self, state: Tensor, action: Tensor) -> Tensor:
        tensor_sa = self.encoder_sa(torch.cat((state, action), dim=1))
        values = torch.concat([decoder_q(tensor_sa) for decoder_q in self.decoder_qs], dim=-1)
        return values  # Q values
