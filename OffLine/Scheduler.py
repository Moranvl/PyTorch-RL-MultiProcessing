from multiprocessing import cpu_count, Pipe, Value
from Config import Config
from .Learner import Learner
from .Worker import Worker


class Scheduler:
    def __init__(self, agent_class, agent_args):
        # define workers and learners
        self.workers = None
        self.learner = None
        self.agent_class = agent_class
        self.share_model_id = Value("L", 0)
        self.init_workers_and_leanrers(agent_args)

    def init_workers_and_leanrers(self, args):
        # compute numbers
        worker_num = Config.WORKER_NUM
        worker_num = worker_num if worker_num != 0 else cpu_count() - 2
        workers_id = list(range(worker_num))

        # init the pipes
        pipes = [Pipe() for _ in range(worker_num)]
        recv_handlers, send_handlers = list(zip(*pipes))

        # TODO: over init
        learner_agent = self.agent_class(
            share_model_id=self.share_model_id, role='learner', agent_args=args
        )
        self.learner = Learner(
                agent=learner_agent, recv_handlers=recv_handlers,
                workers_id=workers_id, share_model_id=self.share_model_id,
        )
        worker_agents = [
            self.agent_class(share_model_id=self.share_model_id, role='learner', agent_args=args)
            for _ in range(worker_num)
        ]
        self.workers = [
            Worker(
                agent=worker_agents[i], send_handler=send_handlers[i], worker_id=i, share_model_id=self.share_model_id,
            )
            for i in range(worker_num)
        ]

    def close(self):
        # terminate
        [p.terminate() for p in self.workers]
        self.learner.terminate()
        # close
        [p.close() for p in self.workers]
        self.learner.close()









