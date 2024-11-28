import datetime
import keyboard
from pathlib import Path
from multiprocessing import cpu_count, Pipe, Value, Process, Event
from Config import Config
from .Learner import Learner
from .Worker import Worker


class Scheduler:
    def __init__(self, agent_class, agent_args, envs):
        # define workers and learners
        self.workers = None
        self.learner = None
        self.models_train_path = None
        self.models_backup_path = None
        self.conn = None

        self.agent_class = agent_class
        self.share_model_id = Value("L", 0)
        self.is_running = Event()
        self.is_running.set()

        self.init_workers_and_leanrers(agent_args, envs)

    def init_workers_and_leanrers(self, args, envs):
        # initialize model folder
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
        self.models_train_path = Path.cwd() / "Models" / now / "train"
        self.models_backup_path = Path.cwd() / "Models" / now / "backup"
        log_path = Path.cwd() / "logs"/ now
        # create the folder to save models
        self.models_train_path.mkdir(parents=True, exist_ok=True)
        self.models_backup_path.mkdir(parents=True, exist_ok=True)

        # compute numbers
        worker_num = Config.WORKER_NUM
        worker_num = worker_num if worker_num != 0 else cpu_count() - 2
        workers_id = list(range(worker_num))

        # initialize the pipes
        pipes = [Pipe() for _ in range(worker_num)]
        self.conn = pipes
        recv_handlers, send_handlers = list(zip(*pipes))

        #  initialize the agents and their processes
        learner_args = {
            "env": envs[-1],
            "share_model_id": self.share_model_id,
            "agent_args": args,
        }
        learner_init_args = {
            "agent_class":  self.agent_class,
            "learner_args": learner_args,
            "load_path_and_id": Config.LOAD_PATH_ID
        }
        self.learner = Learner(
                recv_handlers=recv_handlers,
                workers_id=workers_id, share_model_id=self.share_model_id,
                models_path=self.models_train_path, log_path=str(log_path),
                is_running=self.is_running, learner_init_args=learner_init_args,
        )
        # self.learner.initialize(
        #     agent_class=self.agent_class, learner_args=learner_args,
        #     load_path_and_id=Config.LOAD_PATH_ID
        # )
        worker_args = [
            {
                "env": envs[i],
                "share_model_id": self.share_model_id,
                "agent_args": args,
            }
            for i in range(worker_num)
        ]
        self.workers = [
            Worker(
                agent_class=self.agent_class, worker_args = worker_args[i],
                send_handler=send_handlers[i], worker_id=i, share_model_id=self.share_model_id,
                models_path=self.models_train_path,
                is_running=self.is_running,
            )
            for i in range(worker_num)
        ]

    def close(self):
        # close learner and workers
        self.is_running.clear()
        # Pipes
        for conns in self.conn:
            recv, send = conns
            recv.close()
            send.close()


    def start_work(self):
        """
        start workers and learners to work
        :return: None
        """
        [worker.start() for worker in self.workers]
        self.learner.start()

    def wait_for_stop(self):
        print(" Pressing <T> to stop workers and learner...")
        while True:
            in_put = input()
            if in_put == "T":
                self.close()
                print("The program is closed......")
                break

def close_process(p: Process):
    p.terminate()
    p.join()
    p.close()









