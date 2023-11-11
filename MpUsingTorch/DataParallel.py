import torch
from torch.multiprocessing import Process, Pipe
from Agents.ReplayBuffer import ReplayBuffer
# import torch.multiprocessing as mp
# import os

"""
args:
agent
agent_args
env
num_steps
num_epochs
num_learn
buffer_args
learn_gpu_id

agent_class:
explore_env
update_params
get_params
update_net

buffer:
add_buffer_item
"""


class Worker(Process):
    def __init__(self, agent, env, worker_id, worker_conn: Pipe, learner_conn: Pipe, args: dict):
        super().__init__()
        self.agent = agent(*args['agent_args'])
        self.env = env
        self.worker_conn = worker_conn
        self.learner_conn = learner_conn
        self.worker_id = worker_id

        self.args = args

    def run(self):
        # print(f"{self.pid}+success")
        pass

    def work(self):
        env = self.env
        torch.set_grad_enabled(False)

        '''init agent'''
        params = self.worker_conn.recv()
        if params is None:
            raise ValueError("no params for agent to update")
        self.agent.update_params(params)

        worker_id = self.worker_id

        '''init buffer'''
        num_steps = self.args['num_steps']

        buffer_items = self.agent.explore_env(env, num_steps)
        self.learner_conn.send((worker_id, buffer_items))
        # close the environment
        env.close() if hasattr(env, 'close') else None


class Learner:
    def __init__(self, num_workers, args):
        self.num_workers = num_workers
        self.args = args
        # init agent
        learnner_agent_args = args['agent_args'] + (args["learn_gpu_id"],)
        self.agent = args['agent'](*learnner_agent_args)
        self.agent.act.share_memory()
        args["buffer_args"] = args["buffer_args"] + (args["learn_gpu_id"],)
        self.buffer = ReplayBuffer(*args["buffer_args"])

        self.sum_reward_list = []

        # init workers
        self.worker_pipes = [Pipe(duplex=False) for _ in range(num_workers)]
        self.learner_pipes = [Pipe(duplex=False) for _ in range(num_workers)]
        worker_args = args.copy()
        worker_args['agent_args'] = args['agent_args'] + (-1,)
        self.workers = [
            Worker(
                agent=args['agent'], env=args['env'][worker_id], worker_id=worker_id,
                worker_conn=self.worker_pipes[worker_id][0], learner_conn=self.learner_pipes[worker_id][1],
                args=worker_args,
            )
            for worker_id in range(num_workers)
        ]
        self.worker_start()

        # """Don't set method='fork' when send tensor in GPU"""
        # method = 'spawn' if os.name == 'nt' else 'forkserver'  # os.name == 'nt' means Windows NT operating system (WinOS)
        # mp.set_start_method(method=method, force=True)
        # self.agent.act.share_memory()
        # [worker.start() for worker in self.workers]
        # [worker.join() for worker in self.workers]

    def learn(self):
        torch.set_grad_enabled(True)
        self.agent.update_net(self.buffer)

    def worker_start(self):
        print("worker starting")
        [worker.start() for worker in self.workers]
        [worker.join() for worker in self.workers]
        print("worker start successful")

    def processing_result(self, input_result):
        for pid, buffer_item in input_result:
            self.buffer.update(buffer_item)
            self.sum_reward_list.append(buffer_item[2].sum())

    def explore(self):
        agent = self.agent
        worker_pipes = [conns[1] for conns in self.worker_pipes]
        learner_pipes = [conns[0] for conns in self.learner_pipes]
        workers = self.workers
        for epoch in range(int(self.args['num_epochs'])):
            # update parameters
            params = agent.get_params()
            [conn.send(params) for conn in worker_pipes]
            # explore
            [worker.work() for worker in workers]
            [worker.join() for worker in workers]
            # get result
            result = [conn.recv() for conn in learner_pipes]
            self.processing_result(result)
            self.learn()
            self.output_result()

    def output_result(self):
        len_reward = len(self.sum_reward_list)
        for i in range(len_reward - self.num_workers, len_reward):
            print(f"epoch {i}: {self.sum_reward_list[i]}")

    def close(self):
        for conns in self.learner_pipes + self.worker_pipes:
            for conn in conns:
                conn.close()
        for worker in self.workers:
            worker.close()
