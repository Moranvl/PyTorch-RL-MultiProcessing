import torch
import math
import numpy as np
from copy import deepcopy
from numpy import floating
from torch import Tensor
from torch.nn import Parameter, Module
from torch.utils.tensorboard import SummaryWriter
from .AgentBase import AgentBaseOnLine
from .ReplayBuffer import OnlineReplayBuffer
from .utils import build_mlp, layer_init_with_orthogonal, optimizer_update, soft_update
from Config import Config

args = {
    # envs
    "action_space": {
        "high":     [1,],
        "low":      [-1,],
    },
    # PPO
    "lambda_gae_adv":   0.95,       # could be 0.80~0.99
    "lambda_entropy":   0.001,      # could be 0.00~0.10
    "ratio_clip":       0.2,
    "KL_PPO":           False,
    "d_targ":           0.01,        # 0.003-0.03
    "beta":             1,           # 0.3-10

    "state_dim":        8,
    "action_dim":       8,
    "horizon_len":      200,
    "repeat_times":     25,

    "net_dims": {
        "act": [256, 256, 256],
        "cri": [256, 256, 256],
    },

    "learning_rate": {
        "act":      1e-4,
        "cri":      3e-4,
    },  # the learning rate for network updating
    "update_tau":   0.05,

    "gamma":            0.96,
    "batch_size":       128,
    "clip_grad_norm":   3.0,  # 0.1 ~ 4.0, clip the gradient after normalization
    "buffer":{
        "max_size": int(1e6),
    }
}

class AgentPPO(AgentBaseOnLine):
    """
    continuous.
    """
    def __init__(self, env, role: str, agent_args: dict):
        super().__init__(env, role, agent_args)
        self.exploit_step = 0
        # self.KL_PPO = agent_args['KL_PPO']
        self.state_dim = agent_args["state_dim"]
        self.action_dim = agent_args["action_dim"]
        self.horizon_len = agent_args["horizon_len"]
        self.worker_num = Config.WORKER_NUM
        self.gamma = agent_args['gamma']
        self.repeat_times = agent_args["repeat_times"]
        self.batch_size = agent_args["batch_size"]
        self.soft_update_tau = agent_args["update_tau"]

        self.lambda_gae_adv = agent_args['lambda_gae_adv']
        self.ratio_clip = agent_args['ratio_clip']
        lambda_entropy = agent_args['lambda_entropy']
        self.lambda_entropy = torch.tensor(lambda_entropy, dtype=torch.float32, device=self.device)

        # Policy
        limit = {"high": agent_args["action_space"]["high"], "low": agent_args["action_space"]["low"]}
        self.act = ActorPPO(
            net_dims=agent_args["net_dims"]["act"],
            state_dim=agent_args["state_dim"], action_dim=agent_args["action_dim"],
            device=self.device, limit=limit
        ).to(self.device)
        if role == "learner":
            self.cri = CriticPPO(
                net_dims=agent_args["net_dims"]["cri"],
                state_dim=agent_args["state_dim"], action_dim=agent_args["action_dim"]
            ).to(self.device)
            self.cri_target = deepcopy(self.cri)
            self.criterion = torch.nn.SmoothL1Loss(reduction="mean")
            self.act_optimizer = torch.optim.AdamW(self.act.parameters(), agent_args["learning_rate"]["act"])
            self.cri_optimizer = torch.optim.AdamW(self.cri.parameters(), agent_args["learning_rate"]["cri"])
            self.clip_grad_norm = agent_args["clip_grad_norm"]

            if agent_args["KL_PPO"]:
                self.d_targ = agent_args["d_targ"]
                self.beta = agent_args["beta"]

        # ReplayBuffer
        self.replay_buffer = OnlineReplayBuffer(self.device)

    def update_params(self, params):
        """
        update params for actor.
        :param params: for actor
        :return: None
        """
        self.act.load_state_dict(params)

    def get_params(self):
        """
        get params for actor.
        :return: None
        """
        state_dict = self.act.state_dict()
        state_dict = {k: v.cpu() for k, v in state_dict.items()}
        return state_dict

    def explore(self):
        """
        explore all the env
        :return:  <s, a, r, s'> (to self.update_buffer)
        """
        horizon_len = self.horizon_len
        env = self.env
        get_action = self.act.get_action
        convert_action_for_env = self.act.convert_action_for_env

        states      = torch.zeros((horizon_len + 1, self.state_dim), dtype=torch.float32).to(self.device)
        actions     = torch.zeros((horizon_len, self.action_dim), dtype=torch.float32).to(self.device)  # different
        rewards     = torch.zeros((horizon_len, 1), dtype=torch.float32).to(self.device)
        dones       = torch.zeros((horizon_len, 1), dtype=torch.bool).to(self.device)
        logprobs    =  torch.zeros((horizon_len, 1), dtype=torch.float32).to(self.device)  # different

        state, _ = env.reset()
        state = states[0] = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        for t in range(horizon_len):
            with torch.no_grad():
                action, logprob = get_action(state)

            # ary_action = action.detach().cpu().numpy().item()
            ary_action = convert_action_for_env(action).detach().cpu().numpy()[0]
            ary_state, reward, terminated, truncated, _ = env.step(ary_action)  # next_state
            done = terminated or truncated
            state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            states[t + 1]   = state
            actions[t]      = action
            logprobs[t]     = logprob
            rewards[t]      = reward
            dones[t]        = done

            if done:
                break

        undones = 1.0 - dones.type(torch.float32)

        return states, actions, logprobs, rewards, undones

    def exploit(self, writter: SummaryWriter or None):
        """
        exploit all the env
        :type writter: object
        :return:  kpi
        """
        horizon_len = self.horizon_len
        env = self.env
        get_action = self.act.get_action_exploit
        convert_action_for_env = self.act.convert_action_for_env

        rewards = torch.zeros((horizon_len, 1), dtype=torch.float32).to(self.device)

        state, _ = env.reset()
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        for t in range(horizon_len):
            with torch.no_grad():
                action = get_action(state)

            # ary_action = action.detach().cpu().numpy().item()
            ary_action = convert_action_for_env(action).detach().cpu().numpy()[0]
            ary_state, reward, terminated, truncated, _ = env.step(ary_action)  # next_state
            done = terminated or truncated
            state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            rewards[t] = reward

            if done:
                break
        sum_reward = rewards.sum()
        if writter is not None:
            writter.add_scalar(tag="exploit/sum_reward", scalar_value=sum_reward, global_step=self.exploit_step)
            self.exploit_step += 1
        return sum_reward

    def learn(self, writter: SummaryWriter or None):
        """
        Update the policy of the agent
        :return: None
        """
        buffer = self.replay_buffer.sample()
        buffer_size = buffer[0].shape[0]

        '''get advantages reward_sums'''
        with torch.no_grad():
            states, next_states, actions, logprobs, rewards, undones = buffer
            # set a smaller 'batch_size' to avoid CUDA OOM
            bs = max(1, 2 ** 10 // self.worker_num)
            values = [self.cri_target(states[i:i + bs]) for i in range(0, buffer_size, bs)]
            # values.shape == (buffer_size, 1)
            values = torch.cat(values, dim=0)

            next_values = [self.cri_target(next_states[i:i + bs]) for i in range(0, buffer_size, bs)]
            # values.shape == (buffer_size, 1)
            next_values = torch.cat(next_values, dim=0)

            # shape == (buffer_size, 1)
            advantages = self.get_advantages(rewards, undones, values, next_values)
            reward_sums = advantages + values  # reward_sums.shape == (buffer_size, )
            sum_reward_avg = rewards.sum() / self.worker_num
            del rewards, undones, values

            advantages = (advantages - advantages.mean()) / (advantages[::4, ::4].std() + 1e-5)  # avoid CUDA OOM
            # assert logprobs.shape == advantages.shape == reward_sums.shape == (buffer_size, states.shape[1])
        buffer = states, actions, logprobs, advantages, reward_sums

        '''update network'''
        obj_entropies = []
        obj_critics = []
        obj_actors = []

        torch.set_grad_enabled(True)
        update_times = int(buffer_size * self.repeat_times / self.batch_size)
        assert update_times >= 1
        for update_t in range(update_times):
            obj_critic, obj_actor, obj_entropy = self.update_objectives(buffer, update_t)
            obj_entropies.append(obj_entropy)
            obj_critics.append(obj_critic)
            obj_actors.append(obj_actor)
        torch.set_grad_enabled(False)

        obj_entropy_avg = np.array(obj_entropies).mean() if len(obj_entropies) else 0.0
        obj_critic_avg = np.array(obj_critics).mean() if len(obj_critics) else 0.0
        obj_actor_avg = np.array(obj_actors).mean() if len(obj_actors) else 0.0

        if writter is not None:
            writter.add_scalar(tag="loss/cri", scalar_value=obj_critic_avg, global_step=self.exploit_step)
            writter.add_scalar(tag="loss/act", scalar_value=obj_actor_avg, global_step=self.exploit_step)
            writter.add_scalar(tag="loss/entropy", scalar_value=obj_entropy_avg, global_step=self.exploit_step)
            writter.add_scalar(tag="train/sum_reward_avg", scalar_value=sum_reward_avg, global_step=self.exploit_step)

    def update_objectives(self, buffer_item, update_t: int) -> tuple[floating, floating, floating]:
        states, actions, logprobs, advantages, reward_sums = buffer_item
        sample_len = states.shape[0]

        # random choice.
        # ids = torch.randint(sample_len, size=(self.batch_size,), requires_grad=False, device=self.device)
        #
        # state       = states[ids]
        # action      = actions[ids]
        # logprob     = logprobs[ids]
        # advantage   = advantages[ids]
        # reward_sum  = reward_sums[ids]

        # mini-batch
        objs_critic, objs_surrogate, objs_entropy = [[] for _ in range(3)]
        mini_batch = self.batch_size
        for i in range(0, sample_len, mini_batch):
            state       = states[i:i + mini_batch]
            action      = actions[i:i + mini_batch]
            logprob     = logprobs[i:i + mini_batch]
            advantage   = advantages[i:i + mini_batch]
            reward_sum  = reward_sums[i:i + mini_batch]

            # critic network predicts the reward_sum (Q value) of state
            value = self.cri(state)
            obj_critic = self.criterion(value, reward_sum)
            optimizer_update(self.cri_optimizer, obj_critic, self.clip_grad_norm)
            soft_update(self.cri_target, self.cri, self.soft_update_tau)

            new_logprob, entropy = self.act.get_logprob_entropy(state, action)
            ratio = (new_logprob - logprob.detach()).exp()

            surrogate1 = advantage * ratio
            surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            surrogate = torch.min(surrogate1, surrogate2)  # save as below
            # surrogate = advantage * ratio * torch.where(advantage.gt(0), 1 - self.ratio_clip, 1 + self.ratio_clip)

            obj_surrogate = surrogate.mean()    # major actor objective
            obj_entropy = entropy.mean()        # minor actor objective
            obj_actor_full = obj_surrogate + obj_entropy * self.lambda_entropy
            obj_actor_full = -obj_actor_full
            optimizer_update(self.act_optimizer, obj_actor_full, self.clip_grad_norm)

            # KL: Tensor = logprob.exp() * (logprob - new_logprob)
            # loss_KL: Tensor = -(advantage * ratio -  self.beta * KL)
            # if KL < self.d_targ /1.5:
            #     self.beta /= 2
            # elif KL > self.d_targ * 1.5:
            #     self.beta *= 2
            # optimizer_update(self.act_optimizer, loss_KL, self.clip_grad_norm)

            objs_critic.append(obj_critic.item())
            objs_surrogate.append(obj_surrogate.item())
            objs_entropy.append(obj_entropy.item())

        return np.mean(objs_critic), np.mean(objs_surrogate), np.mean(objs_entropy)

    def update_buffer(self, data) -> None:
        """
        update the new data to buffer
        :param data: new data from the self.explore
        :return:  None
        """
        self.replay_buffer.update_buffer(data)

    def get_advantages(self, rewards: Tensor, undones: Tensor, values, next_values):
        advantages = torch.empty_like(values)  # advantage value

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        # last advantage value by GAE (Generalized Advantage Estimate)
        advantage = torch.zeros_like(values[0])
        # the difference between is the next value.

       # get advantage value using the estimated value of critic network
       #  for t in range(horizon_len - 1, -1, -1):
       #      advantages[t] = rewards[t] + masks[t] * advantage - values[t]
       #      advantage = values[t] + self.lambda_gae_adv * advantages[t]
        for t in range(horizon_len - 1, -1, -1):
            advantages[t] = rewards[t]  - values[t] + masks[t] * (next_values[t] + self.lambda_gae_adv * advantage)
            advantage = advantages[t]
        return advantages

class AgentDiscretePPO(AgentPPO):
    def __init__(self, env, role: str, agent_args: dict):
        super().__init__(env, role, agent_args)
        self.action_dim = 1


        self.act = ActorDiscretePPO(
            net_dims=agent_args["net_dims"]["act"],
            state_dim=agent_args["state_dim"], action_dim=agent_args["action_dim"],
            device=self.device,
        ).to(self.device)
        if role == 'learner':
            self.act_optimizer = torch.optim.AdamW(self.act.parameters(), agent_args["learning_rate"]["act"])


'''network'''


class ActorPPO(Module):
    def __init__(self, net_dims: list[int], state_dim: int, action_dim: int, limit: dict, device):
        super().__init__()
        self.net = build_mlp(dims=[state_dim, *net_dims, action_dim])
        layer_init_with_orthogonal(self.net[-1], std=0.1)

        self.action_std_log = Parameter(torch.zeros((1, action_dim)), requires_grad=True)  # trainable parameter
        # self.ActionDist = torch.distributions.normal.Normal
        self.normal = torch.distributions.normal.Normal(
            loc=torch.tensor(0.0, device=device), scale=torch.tensor(1.0, device=device)
        )
        self.entropy_base = 0.5 + 0.5 * math.log(2 * math.pi)

        limit_high = torch.tensor(limit["high"])
        limit_low = torch.tensor(limit["low"])
        self.center = ((limit_high + limit_low) / 2).to(device)
        self.scale = ((limit_high - limit_low) / 2).to(device)

    def forward(self, state: Tensor) -> Tensor:
        action = self.net(state)
        return self.convert_action_for_env(action)

    def get_action(self, state: Tensor) -> tuple[Tensor, Tensor]:  # for exploration
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        action, logprob = self.resample(action_avg, action_std)
        # logprob = logprob.sum(1)
        return action, logprob

    def get_action_exploit(self, state: Tensor) -> Tensor:  # for exploration
        action = self.net(state)

        return action

    def get_logprob_entropy(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        _, logprob = self.desample(action, loc=action_avg, scale=action_std)
        entropy = self.entropy(action_avg, action_std).sum(1)
        return logprob, entropy

    def entropy(self, loc:Tensor, scale: Tensor) -> Tensor:
        std = torch.zeros_like(loc) + scale
        return self.entropy_base + torch.log(std)

    def resample(self, loc: Tensor, scale: Tensor):
        shape = loc.shape
        eps = self.normal.sample(shape)
        log_prob = self.normal.log_prob(eps)
        return loc + eps * scale, log_prob

    def desample(self, action: Tensor, loc: Tensor, scale: Tensor):
        eps = (action - loc) / scale
        log_prob = self.normal.log_prob(eps)
        return eps, log_prob

    def convert_action_for_env(self, action: Tensor) -> Tensor:
        action = action.tanh()
        return self.scale * action + self.center

class ActorDiscretePPO(Module):
    def __init__(self, net_dims: list[int], state_dim: int, action_dim: int, device):
        super().__init__()
        self.net = build_mlp(dims=[state_dim, *net_dims, action_dim])
        layer_init_with_orthogonal(self.net[-1], std=0.1)

        self.ActionDist = torch.distributions.Categorical
        self.soft_max = torch.nn.Softmax(dim=-1)

    def forward(self, state: Tensor) -> Tensor:
        a_prob = self.net(state)  # action_prob without softmax
        return a_prob.argmax(dim=1)  # get the indices of discrete action

    def get_action(self, state: Tensor) -> (Tensor, Tensor):
        a_prob = self.soft_max(self.net(state))
        a_dist = self.ActionDist(a_prob)
        action = a_dist.sample()
        logprob = a_dist.log_prob(action)
        return action, logprob

    def get_action_exploit(self, state: Tensor) -> Tensor:
        a_prob = self.net(state)  # action_prob without softmax
        return a_prob.argmax(dim=1)  # get the indices of discrete action

    def get_logprob_entropy(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        a_prob = self.soft_max(self.net(state))  # action.shape == (batch_size, 1), action.dtype = th.int
        dist = self.ActionDist(a_prob)
        # logprob = dist.log_prob(action)
        # entropy = dist.entropy()
        logprob = dist.log_prob(action.squeeze()).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        return logprob, entropy
    @staticmethod
    def convert_action_for_env(action: Tensor) -> Tensor:
        return action.long()


class CriticPPO(Module):
    def __init__(self, net_dims: list[int], state_dim: int, action_dim: int, num_ensembles: int = 4):
        super().__init__()
        assert isinstance(action_dim, int)
        self.nets = []
        # self.net = build_mlp(dims=[state_dim, *net_dims, 1])
        # layer_init_with_orthogonal(self.net[-1], std=0.5)

        for net_i in range(num_ensembles):
            net = build_mlp(dims=[state_dim, *net_dims, 1])
            layer_init_with_orthogonal(net[-1], std=0.5)

            self.nets.append(net)
            setattr(self, f"nets{net_i:02}", net)

    def forward(self, state: Tensor) -> Tensor:
        # value = self.net(state)
        values = torch.concat([net(state) for net in self.nets], dim=-1)
        value = values.mean(dim=-1, keepdim=True)
        return value  # advantage value