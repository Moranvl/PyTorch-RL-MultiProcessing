import gym
import pathlib
import shutil
from OffLine import Scheduler
from Config import Config
from torch.multiprocessing import cpu_count

def create_env(env_name):
    envs_num = Config.WORKER_NUM
    envs_num = envs_num if envs_num != 0 else cpu_count() - 2
    envs_num += 1
    envs = [gym.make(env_name) for _ in range(envs_num)]
    return envs

def clear_history():
    local = pathlib.Path.cwd()
    delete_path = [local / "Models", local / "logs"]
    for dp in delete_path:
        if dp .exists() and dp.is_dir():
            for item in dp.iterdir():
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            print("Clear Success")
        else:
            print("Clear Error")

def test_DQN():
    from Agents.AgentDQN import AgentDQN, args
    es = create_env("CartPole-v1")
    env = es[0]
    args["state_dim"], args["action_dim"] = env.observation_space.shape[0], env.action_space.n

    scheduler = Scheduler(agent_class=AgentDQN, agent_args=args, envs=es)
    scheduler.start_work()
    # time.sleep(5)
    # scheduler.close()
    # scheduler.learner.terminate()
    # scheduler.learner.close()
    scheduler.wait_for_stop()

def test_SAC():
    from Agents.AgentSAC import AgentSAC, args
    es = create_env("Pendulum-v1")
    env = es[0]

    args["state_dim"], args["action_dim"] = env.observation_space.shape[0], env.action_space.shape[0]
    args["action_space"]["high"], args["action_space"]["low"] = [2], [-2]
    args["batch_size"] = 256

    scheduler = Scheduler(agent_class=AgentSAC, agent_args=args, envs=es)
    scheduler.start_work()
    scheduler.wait_for_stop()

def test_PPO():
    from Agents.AgentPPO import args as agent_args
    from Agents.AgentPPO import AgentPPO
    from OnLine.Learner import Learner
    agent = AgentPPO
    # for continuous
    es = create_env("Pendulum-v1")
    env = es[0]

    agent_args["state_dim"], agent_args["action_dim"] = env.observation_space.shape[0], env.action_space.shape[0]
    agent_args["action_space"]["high"], agent_args["action_space"]["low"] = [2], [-2]
    agent_args["batch_size"] = 128
    agent_args["repeat_times"] = 4
    agent_args["learning_rate"]["act"], agent_args["learning_rate"]["cri"] = 1e-5, 3e-5
    learner = Learner(AgentPPO, agent_args, es)
    learner.explore()

def test_Discrete_PPO():
    from Agents.AgentPPO import args as agent_args
    from Agents.AgentPPO import AgentDiscretePPO
    from OnLine.Learner import Learner
    agent = AgentDiscretePPO
    # for Discrete
    es = create_env("CartPole-v1")
    env = es[0]

    agent_args["state_dim"], agent_args["action_dim"] = env.observation_space.shape[0], env.action_space.n
    agent_args["batch_size"] = 128
    agent_args["gamma"] = 0.98
    agent_args["lambda_gae_adv"] = 0.85
    agent_args["repeat_times"] = 4
    agent_args["lambda_entropy"] = 0.001
    agent_args["ratio_clip"] = 0.25
    agent_args["learning_rate"]["act"], agent_args["learning_rate"]["cri"] = 1e-4, 3e-4
    learner = Learner(agent, agent_args, es)
    learner.explore()

if __name__ == "__main__":
    if Config.CLEAR_HISTORY:
        clear_history()
    # test_DQN()
    # test_SAC()
    # test_PPO()
    test_Discrete_PPO()

