import time
import datetime
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pipe
from torch.utils.tensorboard import SummaryWriter
from Config import Config
from .Worker import Worker

class Learner:
    def __init__(self, agent_class, agent_args, envs):
        num_workers = Config.WORKER_NUM
        self.num_workers = num_workers

        # init agent
        self.agent = agent_class(env=envs[-1], role='learner', agent_args=agent_args)

        self.sum_reward_list = []
        self.writer = None

        # init workers
        worker_pipes = [Pipe(duplex=False) for _ in range(num_workers)]
        learner_pipes = [Pipe(duplex=False) for _ in range(num_workers)]
        self.workers = [
            Worker(
                env = envs[worker_id], agent_class=agent_class, worker_args=agent_args, worker_id=worker_id,
                # worker is used to recv, learner is used to send
                worker_conn=worker_pipes[worker_id][0], learner_conn=learner_pipes[worker_id][1],
            )
            for worker_id in range(num_workers)
        ]
        self.worker_pipes = [conns[1] for conns in worker_pipes]
        self.learner_pipes = [conns[0] for conns in learner_pipes]

        self.worker_init()

    def learn(self):
        self.agent.learn(self.writer)

    def worker_init(self):
        print("worker starting")

        [worker.start() for worker in self.workers]
        [worker.join() for worker in self.workers]

        # initialize the writer
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
        log_path = Path.cwd() / "logs" / now
        self.writer = SummaryWriter(str(log_path))

        print("worker start successful")

    def explore(self):
        worker_pipes = self.worker_pipes
        workers = self.workers

        '''initialize'''
        [worker.initialize() for worker in workers]

        try:
            while True:
                # update parameters
                params = self.get_act_params()
                [conn.send(params) for conn in worker_pipes]

                # explore
                [worker.explore() for worker in workers]
                self.exploit()
                [worker.join() for worker in workers]

                # get result
                self.get_data()
                self.learn()
                # print sum reward
                # self.output_result()
        except KeyboardInterrupt:
            print("Learner Stopped by KeyboardInterrupt\n")
            plt.plot(self.sum_reward_list)
            plt.show()
        finally:
            # clear the resources
            self.close()

    def exploit(self):
        sum_reward = self.agent.exploit(self.writer)
        self.sum_reward_list.append(
            sum_reward.cpu()
        )

    def get_act_params(self):
        return self.agent.get_params()

    def get_data(self):
        for recv_handler in self.learner_pipes:
            while recv_handler.poll():
                worker_id, data = recv_handler.recv()
                self.agent.update_buffer(data)

    def output_result(self):
        len_reward = len(self.sum_reward_list)
        print(f"epoch {len_reward}: {self.sum_reward_list[-1]}")

    def close(self):
        for worker in self.workers:
            worker.close()
        for conn in self.learner_pipes + self.worker_pipes:
            conn.close()
