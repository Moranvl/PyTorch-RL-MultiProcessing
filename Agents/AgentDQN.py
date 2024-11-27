args = {
    "state_dim": 8,
    "action_dim": 8,
    "horizon_len": 200,

    "net_dims": 512,

    "explore_rate": 0.25,
    "gamma": 0.99,
    "repeat_times": 1,
    "batch_size": 64,
    "soft_update_tau": 5e-3,  # 2 ** -8 ~= 5e-3. the tau of soft target update `net = (1-tau)*net + tau*net1`
    "clip_grad_norm": 3.0,  # 0.1 ~ 4.0, clip the gradient after normalization
    "learning_rate": 6e-5,  # the learning rate for network updating
}

class AgentDQN(AgentBase):
    """
    Deep Q-Network algorithm.
    """
    def __init__(self, share_model_id: Value, role: str, agent_args: dict):
        super().__init__(share_model_id, role, agent_args)
        # Args
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

        # Policy
        self.cri = self.act = QNet(agent_args["net_dims"], self.state_dim, self.action_dim).to(self.device)
        self.cri_target = deepcopy(self.cri)
        self.cri_optimizer = self.act_optimizer = torch.optim.AdamW(self.act.parameters(), agent_args["learning_rate"])
        self.act.explore_rate = args["explore_rate"]

    def explore(self):
        # self.explore_env(env, self.horizon_len)
        pass

    def update_buffer(self, data) -> None:
        pass

    def learn(self):
        pass

    def save_model(self, model_id: int):
        pass

    def load_model(self, model_id: int):
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

