from MpUsingTorch.DataParallel import Learner
from Agents.OldAgentDQN import dqn_args, AgentDQN

import matplotlib.pyplot as plt
import gym

if __name__ == "__main__":
    num_workers = 4

    env = gym.make("CartPole-v1")
    observation_n, action_n = env.observation_space.shape[0], env.action_space.n
    del env

    dqn_copy_args = dqn_args.copy()
    dqn_copy_args["batch_size"] = 128
    dqn_copy_args["repeat_times"] = 2
    learner_args = {
        "agent": AgentDQN,
        "agent_args": (dqn_copy_args, [256, 256, 256], observation_n, action_n, ),
        "buffer_args": (int(1e6), observation_n, 1,),
        "env": [gym.make("CartPole-v1") for _ in range(num_workers)],
        "num_epochs": 200,
        "num_steps": 200,
        "learn_gpu_id": 0,
    }
    dqn_args.copy()
    learner = Learner(num_workers=num_workers, args=learner_args)
    learner.explore()
    plt.plot(learner.sum_reward_list)
    plt.show()
    # learner.worker_start()
