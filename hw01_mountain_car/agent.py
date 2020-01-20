import random
import numpy as np
import os
from .train import transform_state


class Agent:
    def __init__(self):
        self.weight, self.bias = np.load(__file__[:-8] + "/agent.npz")
        
    def act(self, state):
        return np.argmax(self.weight.dot(transform_state(state)) + self.bias)

    def reset(self):
        pass

