import time
from multiprocessing import Process, Condition, Value
from Agents.AgentBase import AgentBase
from Config import Config


class Worker(Process):
    """
    Worker
    """
    def __init__(
            self, agent: AgentBase,
            send_handler: Condition,
            worker_id: int, share_model_id: Value,
    ):
        super().__init__()
        self.worker_id = worker_id
        self.agent = agent
        self.sleep_time = Config.WORKER_SLEEP_TIME
        self.send_handler = send_handler
        self.share_model_id: Value = share_model_id

    def run(self):
        while True:
            data = self.explore()
            self.send_handler.send(data)
            time.sleep(self.sleep_time)

    def explore(self):
        return self.agent.explore()
