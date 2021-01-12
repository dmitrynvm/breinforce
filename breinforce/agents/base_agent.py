
class BaseAgent:

    def __init__(self, actions=None):
        self.actions = actions

    def act(self, obs):
        raise NotImplementedError()
