from Agents.AgentDQN import AgentDQN, dqn_args
from Agents.ReplayBuffer import ReplayBuffer
import gym

if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    observation_n, action_n = env.observation_space.shape[0], env.action_space.n

    args = dqn_args.copy()
    args["repeat_times"] = 1
    args["batch_size"] = 128

    agent = AgentDQN(
        args, net_dims=[256, 256, 256], state_dim=observation_n, action_dim=action_n, gpu_id=0
    )
    buffer = ReplayBuffer(
        max_size=int(1e6),
        state_dim=observation_n,
        # 离散
        action_dim=1,
        gpu_id=0,
    )

    sum_rewards = 0
    for i in range(100):
        buffer_item = agent.explore_env(env, 200)
        sum_rewards = buffer_item[2].sum()
        buffer.update(buffer_item)
        agent.update_net(buffer)
        print(f"epoch: {i}, sum reward: {sum_rewards}")



