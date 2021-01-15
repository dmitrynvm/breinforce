import random
from .base_agent import BaseAgent


class RandomBaseAgent(BaseAgent):
    def __init__(self):
        super().__init__()

    def act(self, obs):
        return random.randint(obs["min_raise"], obs["max_raise"])
