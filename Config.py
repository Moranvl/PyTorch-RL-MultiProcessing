from Agents.AgentBase import AgentBase


class Config:
    # global
    AgentClass = AgentBase
    CLEAR_HISTORY = True

    # Scheduler
    """the num of workers and learners, 0 means all computing resources"""
    WORKER_NUM: int = 8
    MODEL_POOL_SIZE: int = 20

    # Worker
    WORKER_SLEEP_TIME = 0.02    # seconds

    # Learner
    LEARNER_SLEEP_TIME = 0.02      # seconds
    LOAD_PATH_ID = None
    LEARNER_SAVE_MODEL_TIME = 10    # seconds
    # path should be out of the id foler
    # LOAD_PATH_ID = (Path.cwd() / "Models" / "now" / "backup", 234)
