import random
import numpy as np
import os
import torch

BATCH_SIZE = 128
BETA = 50


class Exploration:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def get_exploration_reward(states, actions, next_states):
        return 0 # TODO, should return tensor of size `len(actions)`
        
    def update(transition)
        state, action, next_state, reward, done = transition


class Agent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim # dimensionalite of state space
        self.action_dim = action_dim # count of available actions
        self.exploration = Exploration(state_dim, action_dim)
        self.memory = deque(maxlen=20000)
        
    def act(self, state):
        return 0 # TODO

    def update(self, transition):
        state, action, next_state, reward, done = transition
        self.exploration.updatae(transition)
        if len(self.memory) < BATCH_SIZE:
            return
        state, action, next_state, reward, done = zip(*random.sample(self.memory, BATCH_SIZE))
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action)
        next_state = torch.tensor(next_state, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float) + BETA * self.exploration.get_exploration_reward(state, action, next_state)        

    def reset(self):
        pass

