import time
import datetime
import queue
import shutil
from pathlib import Path
from typing import List, Tuple
from multiprocessing import Process, Condition, Value
from Agents.AgentBase import AgentBase
from Config import Config


class Learner(Process):
    def __init__(
            self, agent: AgentBase,
            recv_handlers: Tuple[Condition],
            workers_id: List[int], share_model_id: Value,
    ):
        super().__init__()
        self.workers_id = workers_id
        self.agent = agent
        self.recv_handlers = recv_handlers
        self.share_model_id: Value = share_model_id

        # initialize model folder
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
        self.models_path = Path.cwd() / "Models" / now
        # create the folder to save models
        self.models_path.mkdir(parents=True, exist_ok=True)
        # create the model pool
        self.model_pool = ModelPool(self.models_path)

        self.sleep_time = Config.LEARNER_SLEEP_TIME

    def run(self):
        while True:
            self.get_data()
            self.learn()
            self.save_model()
            time.sleep(self.sleep_time)

    def learn(self):
        self.agent.learn()

    def get_data(self):
        for recv_handler in self.recv_handlers:
            while recv_handler.poll():
                data = recv_handler.recv()
                self.agent.update_buffer(data)

    def save_model(self):
        model_id = self.model_pool.save_model()
        self.agent.save_model(model_id)
        # update the share memory
        with self.share_model_id.get_lock():
            self.share_model_id.value = model_id


class ModelPool:
    """
    Manages the models.
    """

    def __init__(self, model_path):
        self.model_path: Path = model_path
        # init the model pool
        self.max_size = Config.MODEL_POOL_SIZE
        self.model_id_pool = queue.Queue(maxsize=Config.MODEL_POOL_SIZE)
        [self.model_id_pool.put(0) for _ in range(Config.MODEL_POOL_SIZE)]
        # create the id
        self.latest_id: int = 0

    def save_model(self):
        self.latest_id = self.latest_id + 1 if self.latest_id<self.max_size else 1
        self.update_model_id_pool(self.latest_id)
        (self.model_path/f"{self.latest_id}").mkdir(parents=True, exist_ok=True)
        return self.latest_id

    def update_model_id_pool(self, model_id):
        delete_id = self.model_id_pool.get()
        self.model_id_pool.put(model_id)
        # delete the model until it is really be there.
        if delete_id != 0:
            shutil.rmtree(self.model_path/f"{delete_id}")
