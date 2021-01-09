
class BaseAgent:

    def __init__(self, legal_actions=None):
        self.legal_actions = legal_actions

    def act(self, obs):
        raise NotImplementedError()
