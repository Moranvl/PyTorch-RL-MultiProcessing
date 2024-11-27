from multiprocessing import Value

class AgentBase:
    def __init__(self, share_model_id: Value, role: str, agent_args: dict):
        self.share_model_id: Value = share_model_id
        self.role: str = role

    def explore(self):
        """
        explore all the env
        :return:  <s, a, r, s'> (to self.update_buffer)
        """
        raise NotImplementedError

    def update_buffer(self, data) -> None:
        """
        update the new data to buffer
        :param data: new data from the self.explore
        :return:  None
        """
        raise NotImplementedError

    def learn(self):
        """
        Update the policy of the agent
        :return: None
        """
        raise NotImplementedError

    def save_model(self, model_id: int):
        """
        Save the model of the agent
        :param model_id: the folder of model to save
        :return: None
        """
        raise NotImplementedError

    def load_model(self, model_id: int):
        """
        Save the model of the agent
        :param model_id: the folder of model to load
        :return: None
        """
        raise NotImplementedError
