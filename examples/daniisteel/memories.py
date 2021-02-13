import torch
import random
from collections import namedtuple


Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))


class SequentialMemory():
    def __init__(self, size):
        self.size = size
        self.curr = 0
        self.items = []

    def add(self, item):
        if len(self.items) < self.size:
            self.items.append(item)
        else:
            self.items[self.curr % self.size] = item
        self.curr += 1

    def extract(self, experiences):
        batch = Experience(*zip(*experiences))
        t1 = torch.cat(batch.state)
        t2 = torch.cat(batch.action)
        t3 = torch.stack(batch.reward, axis=0)
        t4 = torch.cat(batch.next_state)
        return (t1, t2, t3, t4)

    def sample(self, n_items):
        return self.extract(random.sample(self.items, n_items))

    def ready(self, n_items):
        return n_items < len(self.items)