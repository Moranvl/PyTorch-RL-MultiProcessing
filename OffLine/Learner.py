import time
import shutil
from pathlib import Path
from typing import List, Tuple
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Process, Condition, Value, Queue, Event
from Config import Config


class Learner(Process):
    def __init__(
            self,
            recv_handlers: Tuple[Condition],
            workers_id: List[int], share_model_id: Value,
            models_path: Path, log_path: str, is_running: Event,
            learner_init_args: dict,
    ):
        super().__init__()
        self.agent = None
        self.model_pool = None
        self.writer = None

        self.workers_id = workers_id
        self.recv_handlers = recv_handlers
        self.share_model_id: Value = share_model_id
        self.is_running = is_running
        self.models_path = models_path
        self.log_path = log_path
        self.learner_init_args = learner_init_args

        self.sleep_time = Config.LEARNER_SLEEP_TIME
        self.save_model_time = Config.LEARNER_SAVE_MODEL_TIME

    def initialize(self, agent_class, learner_args: dict, load_path_and_id: Tuple,):
        # initialize the agent
        learner_agent = agent_class(
            env=learner_args["env"], share_model_id=learner_args["share_model_id"],
            role='learner', agent_args=learner_args["agent_args"],
        )
        self.agent = learner_agent
        # initialize the moelpool
        self.model_pool = ModelPool(self.models_path)
        # whether the agent should initialize from zero
        if load_path_and_id is not None:
            load_path, load_id = load_path_and_id
            self.agent.load_model(model_id=load_id, model_path=load_path, is_for_train=False)

    def run(self):
        learner_init_args = self.learner_init_args
        self.initialize(
            agent_class=learner_init_args["agent_class"], learner_args=learner_init_args["learner_args"],
            load_path_and_id=learner_init_args["load_path_and_id"],
        )

        is_running = self.is_running
        print("Learner Starting\n")
        # initialize the writer
        self.writer = SummaryWriter(self.log_path)
        try:
            now = time.time()
            while is_running.is_set():
                self.get_data()
                if self.is_need_train():
                    self.learn()
                    if time.time() - now > self.save_model_time:
                        self.save_model_for_train()
                        now = time.time()
                    self.agent.exploit(self.writer)
                time.sleep(self.sleep_time)
            print("Learner Closed\n")
        except KeyboardInterrupt:
            print("Learner Stopped by KeyboardInterrupt\n")
        finally:
            # clear the resources
            self.cleanup()

    def learn(self):
        self.agent.learn(self.writer)

    def get_data(self):
        for recv_handler in self.recv_handlers:
            while recv_handler.poll():
                data = recv_handler.recv()
                self.agent.update_buffer(data)

    def is_need_train(self):
        return self.agent.replay_buffer.is_need_train()

    def save_model_for_train(self):
        model_id = self.model_pool.save_model()
        self.agent.save_model(model_id, self.models_path, is_for_train=True)
        # update the share memory
        with self.share_model_id.get_lock():
            self.share_model_id.value = model_id

    def close(self):
        super().close()

    def terminate(self):
        self.cleanup()
        super().terminate()

    def cleanup(self):
        if hasattr(self, "agent"):
            self.agent.close()
            del self.agent
        if hasattr(self, "writer"):
            if isinstance(self.writer, SummaryWriter):
                self.writer.close()
            del self.writer


class ModelPool:
    """
    Manages the models.
    """

    def __init__(self, model_path):
        self.model_path: Path = model_path
        # init the model pool
        self.max_size = Config.MODEL_POOL_SIZE
        self.model_id_pool = Queue(maxsize=Config.MODEL_POOL_SIZE)
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
