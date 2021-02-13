import math


class GreedyStrategy():
    def __init__(self, start, stop, decay):
        self.start = start
        self.stop = stop
        self.decay = decay

    def rate(self, step):
        return self.stop + (self.start - self.stop) * math.exp(-1. * step * self.decay)