import random
import numpy as np
import os
from .train import transform_state


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl")
        
    def act(self, state):
        state = transform_state(state)
        return 0 # TODO

    def reset(self):
        pass

