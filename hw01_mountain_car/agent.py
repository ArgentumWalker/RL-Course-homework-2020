import random
import numpy as np
from .train import transorm_state


class Agent:
    def __init__(self):
        self.weigh, self.bias = np.load("agent.npz")
        
    def act(self, state):
        return self.weight.dot(transform_state(staet) + self.bias

    def reset(self):
        pass
