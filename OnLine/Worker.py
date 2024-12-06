from Agents.AgentBase import AgentBaseOnLine
from multiprocessing import Process, Condition, Value, Event

class Worker(Process):
    def __init__(
            self, env, agent_class, worker_args: dict,
            worker_id: int, worker_conn: Condition, learner_conn: Condition,
    ):
        super().__init__()
        self.agent = None
        self.env = env
        self.agent_args = {"agent_class": agent_class, "worker_args": worker_args}

        self.worker_id = worker_id
        self.worker_conn = worker_conn
        self.learner_conn = learner_conn

        self.worker_id = worker_id

    def initialize(self):
        agent_class = self.agent_args['agent_class']
        worker_args = self.agent_args['worker_args']
        self.agent = agent_class(env=self.env, role='worker', agent_args=worker_args)


    def run(self):
        self.initialize()

    def explore(self):
        """
        explore the env for once.
        :return:
        """
        '''init agent'''
        params = self.worker_conn.recv()
        if params is None:
            raise ValueError("no params for agent to update")
        self.agent.update_params(params)

        '''init buffer'''
        data = self.agent.explore()
        self.learner_conn.send((self.worker_id, data))