import torch
from multiprocessing import Value
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class AgentBase:
    def __init__(self, env, share_model_id: Value, role: str, agent_args: dict):
        self.env = env
        self.share_model_id: Value = share_model_id
        self.role: str = role
        self.replay_buffer = None

        # Devices
        self.device = None
        if role == 'learner' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif role == 'worker' or role == 'learner':
            self.device = torch.device('cpu')
        else:
            raise ValueError(f"role is error! role: {role}")

    def explore(self, model_id: int, model_path: Path):
        """
        explore all the env
        :return:  <s, a, r, s'> (to self.update_buffer)
        """
        raise NotImplementedError

    def exploit(self, writter: SummaryWriter or None):
        """
        exploit the env
        :type writter: tensorboard
        :return:  exploit kpi
        """
        raise NotImplementedError

    def update_buffer(self, data) -> None:
        """
        update the new data to buffer
        :param data: new data from the self.explore
        :return:  None
        """
        raise NotImplementedError

    def learn(self, writter: SummaryWriter or None):
        """
        Update the policy of the agent
        :return: None
        """
        raise NotImplementedError

    def save_model(self, model_id: int, model_path: Path, is_for_train: bool):
        """
        Save the model of the agent
        :type is_for_train: for training
        :param model_path: the path to the model
        :param model_id: the folder of model to save
        :return: None
        """
        raise NotImplementedError

    def load_model(self, model_id: int, model_path: Path, is_for_train: bool):
        """
        Save the model of the agent
        :param is_for_train:  for training
        :param model_path: the path to the model
        :param model_id: the folder of model to load
        :return: None
        """
        raise NotImplementedError

    def close(self):
        """
        close all cuda tensors.
        :return: None
        """
        raise NotImplementedError

class AgentBaseOnLine:
    def __init__(self, env, role: str, agent_args: dict):
        self.env = env
        self.role: str = role
        self.replay_buffer = None

        # Devices
        self.device = None
        if role == 'learner' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif role == 'worker' or role == 'learner':
            self.device = torch.device('cpu')
        else:
            raise ValueError(f"role is error! role: {role}")

    def update_params(self, params):
        """
        update params for actor.
        :param params: for actor
        :return: None
        """
        raise NotImplementedError

    def get_params(self):
        """
        get params for actor.
        :return: None
        """
        raise NotImplementedError

    def explore(self):
        """
        explore all the env
        :return:  <s, a, r, s'> (to self.update_buffer)
        """
        raise NotImplementedError

    def exploit(self, writter: SummaryWriter or None):
        """
        exploit all the env
        :return:  <s, a, r, s'> (to self.update_buffer)
        """
        raise NotImplementedError

    def learn(self, writter: SummaryWriter or None):
        """
        Update the policy of the agent
        :return: None
        """
        raise NotImplementedError

    def update_buffer(self, data) -> None:
        """
        update the new data to buffer
        :param data: new data from the self.explore
        :return:  None
        """
        raise NotImplementedError

