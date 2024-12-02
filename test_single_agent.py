import gym
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from Agents.AgentDQN import AgentDQN, args

def test_agent(environment, agent_args, writer):
    agent = AgentDQN(environment, share_model_id=0, role='learner', agent_args=agent_args)
    for _ in range(200):
        data = agent.explore(0, None)
        print(data[2].sum())
        agent.update_buffer(data)
        agent.learn(writer)
    print(agent.exploit(writer))

def test_agent_commucation(environment, agent_args, writer):
    learn_agent = AgentDQN(environment, share_model_id=0, role='learner', agent_args=agent_args)
    worker_agent = AgentDQN(environment, share_model_id=0, role='worker', agent_args=agent_args)
    model_path = Path.cwd() / "Models" / "test"
    for model_id in range(200):
        data = worker_agent.explore(0, None)

        learn_agent.update_buffer(data)
        learn_agent.learn(writer)

        learn_agent.save_model(model_id, model_path)
        worker_agent.load_model(model_id, model_path)

        print(learn_agent.exploit(writer))

if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    args["state_dim"], args["action_dim"] = env.observation_space.shape[0], env.action_space.n
    tensorboard_writer = SummaryWriter(str(Path.cwd() / "logs" / "test"))
    # test_agent(env, args, tensorboard_writer)
    test_agent_commucation(env, args, tensorboard_writer)


