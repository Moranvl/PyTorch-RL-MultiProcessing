import gym
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from Config import Config
from Agents.AgentDQN import AgentDQN
from Agents.AgentSAC import AgentSAC

def test_agent(environment, agent_args, writer, agent_class):
    rs = []
    agent = agent_class(environment, share_model_id=0, role='learner', agent_args=agent_args)
    for ep in range(2000):
        data = agent.explore(0, None)
        rs.append(data[2].sum().cpu().numpy().item())
        print(f"{ep}:\t{rs[-1]}")
        agent.update_buffer(data)
        agent.learn(writer)
        agent.exploit_step += 1
    print(agent.exploit(writer))
    plt.plot(rs)
    plt.show()

def test_online_agent(environment, agent_args, writer, agent_class):
    rs = []
    num_workers = Config.WORKER_NUM
    learner = agent_class(environment, role='learner', agent_args=agent_args)
    worker = agent_class(environment, role='learner', agent_args=agent_args)
    for ep in range(2000):
        # data = learner.explore()
        data = worker.explore()
        rs.append(data[3].sum().cpu().numpy().item())
        print(f"{ep}:\t{rs[-1]}")
        learner.update_buffer(data)
        learner.learn(writer)
        learner.exploit(writer)
        worker.update_params(learner.get_params())
    # print(agent.exploit(writer))
    plt.plot(rs)
    plt.show()


def test_agent_commucation(environment, agent_args, writer, agent_class):
    learn_agent = agent_class(environment, share_model_id=0, role='learner', agent_args=agent_args)
    worker_agent = agent_class(environment, share_model_id=0, role='worker', agent_args=agent_args)
    model_path = Path.cwd() / "Models" / "test"
    for model_id in range(1000):
        data = worker_agent.explore(0, None)

        learn_agent.update_buffer(data)
        learn_agent.learn(writer)

        learn_agent.save_model(model_id, model_path, is_for_train=True)
        worker_agent.load_model(model_id, model_path, is_for_train=True)

        # print(learn_agent.exploit(writer))
        kpi = learn_agent.exploit(writer)
        print(f"{model_id}:\t{kpi}")

def test_DQN():
    from Agents.AgentDQN import args as agent_args
    from Agents.AgentDQN import AgentDoubleDQN, AgentD3QN
    # agent = AgentDQN
    # agent = AgentDoubleDQN
    agent = AgentD3QN
    # for discrete
    env = gym.make("CartPole-v1")

    agent_args["state_dim"], agent_args["action_dim"] = env.observation_space.shape[0], env.action_space.n
    # agent_args["soft_update_tau"] = 0.05
    # agent_args["batch_size"] = 256
    # agent_args["learning_rate"] = 1e-5

    tensorboard_writer = SummaryWriter(str(Path.cwd() / "logs" / "test"))
    test_agent(env, agent_args, tensorboard_writer, agent)
    # test_agent_commucation(env, args, tensorboard_writer, agent)

def test_SAC():
    from Agents.AgentSAC import args as agent_args
    agent = AgentSAC
    # for continuous
    env = gym.make("Pendulum-v1")

    agent_args["state_dim"], agent_args["action_dim"] = env.observation_space.shape[0], env.action_space.shape[0]
    agent_args["action_space"]["high"], agent_args["action_space"]["low"] = [2], [-2]
    agent_args["batch_size"] = 256
    tensorboard_writer = SummaryWriter(str(Path.cwd() / "logs" / "test"))
    # test_agent(env, agent_args, tensorboard_writer, agent)
    test_agent_commucation(env, agent_args, tensorboard_writer, agent)

def test_PPO():
    from Agents.AgentPPO import args as agent_args
    from Agents.AgentPPO import AgentPPO
    agent = AgentPPO
    # for continuous
    env = gym.make("Pendulum-v1")

    agent_args["state_dim"], agent_args["action_dim"] = env.observation_space.shape[0], env.action_space.shape[0]
    agent_args["action_space"]["high"], agent_args["action_space"]["low"] = [2], [-2]
    agent_args["batch_size"] = 128
    agent_args["learning_rate"]["act"], agent_args["learning_rate"]["cri"] = 1e-5, 3e-5
    tensorboard_writer = SummaryWriter(str(Path.cwd() / "logs" / "test"))
    test_online_agent(env, agent_args, tensorboard_writer, agent)
    # test_agent(env, agent_args, tensorboard_writer, agent)
    # test_agent_commucation(env, agent_args, tensorboard_writer, agent)

def test_Discrete_PPO():
    from Agents.AgentPPO import args as agent_args
    from Agents.AgentPPO import AgentDiscretePPO
    agent = AgentDiscretePPO
    # for continuous
    env = gym.make("CartPole-v1")

    agent_args["state_dim"], agent_args["action_dim"] = env.observation_space.shape[0], env.action_space.n
    agent_args["batch_size"] = 128
    agent_args["gamma"] = 0.98
    agent_args["lambda_gae_adv"] = 0.85
    agent_args["repeat_times"] = 32
    agent_args["lambda_entropy"] = 0.001
    agent_args["ratio_clip"] = 0.25
    agent_args["learning_rate"]["act"], agent_args["learning_rate"]["cri"] = 1e-4, 3e-4
    tensorboard_writer = SummaryWriter(str(Path.cwd() / "logs" / "test"))
    test_online_agent(env, agent_args, tensorboard_writer, agent)


if __name__ == '__main__':
    test_DQN()
    # test_SAC()
    # test_PPO()
    # test_Discrete_PPO()



