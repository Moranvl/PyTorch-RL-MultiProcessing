import time
from pathlib import Path
from multiprocessing import Process, Condition, Value, Event
from Config import Config


class Worker(Process):
    """
    Worker
    """
    def __init__(
            self, agent_class, worker_args: dict,
            send_handler: Condition,
            worker_id: int, share_model_id: Value,
            models_path: Path, is_running: Event,
    ):
        super().__init__()
        # initialize the agent
        worker_agent = agent_class(
            env=worker_args["env"], share_model_id=worker_args["share_model_id"],
            role='worker', agent_args=worker_args["agent_args"],
        )
        self.agent = worker_agent

        self.worker_id = worker_id
        self.models_path = models_path

        self.sleep_time = Config.WORKER_SLEEP_TIME
        self.send_handler = send_handler
        self.share_model_id: Value = share_model_id
        self.is_running = is_running

    def run(self):
        is_running = self.is_running
        print(f"Worker:{self.worker_id} Starting...")
        try:
            while is_running.is_set():
                data = self.explore()
                self.send_handler.send(data)
                time.sleep(self.sleep_time)
            print(f"Worker:{self.worker_id} Closed\n")
        except KeyboardInterrupt:
            print(f"Worker:{self.worker_id} Stopped by KeyboardInterrupt\n")
        finally:
            # clear the resources
            self.cleanup()

    def explore(self):
        model_id, model_path = self.share_model_id.value, self.models_path
        return self.agent.explore(model_id, model_path)

    def cleanup(self):
        pass
