import gym
import pathlib
import shutil
from OffLine import Scheduler
from Agents.AgentDQN import AgentDQN, args
from Config import Config
from torch.multiprocessing import cpu_count

def create_env():
    envs_num = Config.WORKER_NUM
    envs_num = envs_num if envs_num != 0 else cpu_count() - 2
    envs_num += 1
    envs = [gym.make("CartPole-v1") for _ in range(envs_num)]
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


if __name__ == "__main__":
    if Config.CLEAR_HISTORY:
        clear_history()
    es = create_env()
    env = es[0]
    args["state_dim"], args["action_dim"] = env.observation_space.shape[0], env.action_space.n

    scheduler = Scheduler(agent_class=AgentDQN, agent_args=args, envs=es)
    scheduler.start_work()
    # time.sleep(5)
    # scheduler.close()
    # scheduler.learner.terminate()
    # scheduler.learner.close()
    scheduler.wait_for_stop()

# import torch
# from multiprocessing import Pipe, Process
# def test_f(p):
#     tensor = p.recv()
#     print(tensor)
#
# if __name__ == "__main__":
#     recv, send = Pipe()
#     p = Process(target=test_f, args=(recv, ))
#     p.start()
#     send.send(torch.Tensor([1,2,3]))
#     send.send(torch.Tensor([1, 2, 3]))
#     send.close()
#     recv.close()

