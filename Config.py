from Agents.AgentBase import AgentBase


class Config:
    # global
    AgentClass = AgentBase

    # Scheduler
    """the num of workers and learners, 0 means all computing resources"""
    WORKER_NUM: int = 0
    MODEL_POOL_SIZE: int = 20

    # Worker
    WORKER_SLEEP_TIME = 0.02    # seconds

    # Learner
    LEARNER_SLEEP_TIME = 0      # seconds
