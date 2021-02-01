
class BaseAgent(object):

    def __init__(self):
        pass

    def predict(self, obs):
        raise NotImplementedError()
