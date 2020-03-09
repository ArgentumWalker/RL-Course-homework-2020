from gym import make
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import copy
from collections import deque
import random
import time
import json

GAMMA = 0.99
TAU = 0.01
HIDDEN_SIZE = 256
ENV_NAME = "MountainCarContinuous-v2"
device = torch.device("cuda")


def transform_state(state):
    return state


def collect_trajectories(count=2000, agent=None, noise_std=1.):
    env = make(ENV_NAME)
    dataset = TrajectoryDataset()
    for _ in range(count):
        state = env.reset()
        trajectory = []
        done = False
        while not done:
            if agent is not None:
                action = (agent.act(state) + noise_std * np.random.randn(env.action_space.shape[0])).clip(-1, 1)
            else:
                action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            trajectory.append((np.array(state).tolist(), np.array(action).tolist(), np.array(next_state).tolist(), np.array(reward).tolist(), np.array(done).tolist()))
            state = next_state
        dataset.add_trajectory(trajectory)
    return dataset


class TrajectoryDataset(Dataset):
    def __init__(self, sequence_len=16):
        self.trajectories = []
        self.sequence_len = sequence_len

    def add_trajectory(self, trajectory):
        if len(trajectory) < self.sequence_len:
            return
        self.trajectories.append(trajectory)

    def __getitem__(self, item):
        trajectory = self.trajectories[item]
        idx = random.randint(0, len(trajectory) - self.sequence_len)
        return [torch.tensor(x, dtype=torch.float) for x in zip(*trajectory[idx:idx + self.sequence_len])]

    def __len__(self):
        return len(self.trajectories)


class WorldModel:
    def __init__(self, state_dim, action_dim):
        self.input = torch.nn.Sequential(
            torch.nn.Linear(state_dim + action_dim, HIDDEN_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            torch.nn.ReLU()
        )
        self.rnn = torch.nn.LSTM(HIDDEN_SIZE, HIDDEN_SIZE, 3, batch_first=True)
        self.state = torch.nn.Sequential(
            torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_SIZE, state_dim)
        )
        self.reward = torch.nn.Sequential(
            torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_SIZE, 1)
        )
        self.input.to(device)
        self.rnn.to(device)
        self.state.to(device)
        self.reward.to(device)
        self.hidden = None
        self.optim = torch.optim.Adam(list(self.input.parameters()) + list(self.rnn.parameters())
                                      + list(self.state.parameters()) + list(self.reward.parameters()), lr=1e-4)

    def reset(self):
        self.hidden = None

    def predict(self, state, action):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float)
            action = torch.tensor(action, dtype=torch.float)
            x = self.input(torch.cat((state, action), dim=-1).to(device)).unsqueeze(0).unsqueeze(0)
            x, self.hidden = self.rnn(x, self.hidden)
            state, reward = self.state(x), self.reward(x)
        return state.view(-1).cpu().numpy(), reward.item()

    def update(self, batch_sequence):
        state, action, next_state, reward, done = batch_sequence
        state = state.to(device)
        action = action.to(device)
        next_state = next_state.to(device)
        reward = reward.to(device)
        # TODO: Update model


class DDPG:
    def __init__(self, state_dim, action_dim):
        #TODO: Use your implementation of DDPG/TD3/A2C/PPO
        pass

    def update(self, transition):
        pass

    def act(self, state, target=False):
        pass

    def save(self):
        torch.save(self.actor, f"agent.pkl")


class FakeEnv:
    def __init__(self, world_model, start_states, steps_limit=1000):
        self.world_model = world_model
        self.start_states = start_states
        self.steps_limit = steps_limit
        self.steps = 0
        self.state = None

    def step(self, action):
        state, reward= self.world_model.predict(self.state, action)
        self.state = copy.deepcopy(state)
        self.steps += 1
        return state, reward, (self.steps > self.steps_limit), {}

    def reset(self):
        self.state = self.start_states[random.randint(0, len(self.start_states)-1)]
        self.world_model.reset()
        self.steps = 0
        return copy.deepcopy(self.state)


def load_dataset(path):
    trajectories = json.load(open(path, mode="r"))
    dataset = TrajectoryDataset()
    for trajectory in trajectories:
        dataset.add_trajectory(trajectory)
    return dataset

def train_worldmodel(dataset, state_dim, action_dim, epochs=500):
    dataloader = DataLoader(dataset, shuffle=True, batch_size=128)
    wm = WorldModel(state_dim, action_dim)
    for _ in range(epochs):
        t = time.time()
        for batch in dataloader:
            wm.update(batch)
        print("Epoch time:", time.time() - t)
    return wm


def train_agent(train_env, state_dim, action_dim, test_env=None, episodes=250):
    algo = DDPG(state_dim=state_dim, action_dim=action_dim)
    eps = 0.1

    for i in range(episodes):
        state = transform_state(train_env.reset())
        total_reward = 0
        steps = 0
        done = False
        while not done:
            action = (algo.act(state) + np.random.randn(action_dim) * eps).clip(-1, 1)
            next_state, reward, done, _ = train_env.step(action)
            next_state = transform_state(next_state)
            total_reward += reward
            steps += 1
            algo.update((state, action, next_state, reward, done))
            state = next_state

        print("Episode", i)
        print("Exploration", total_reward, steps, eps)
        state = train_env.reset()
        total_reward = 0
        steps = 0
        done = False
        while not done:
            action = algo.act(state, target=True)
            state, reward, done, _ = train_env.step(action)
            total_reward += reward
            steps += 1
        print("Train env", total_reward, steps)

        if test_env is not None:
            state = test_env.reset()
            total_reward = 0
            steps = 0
            done = False
            while not done:
                action = algo.act(state, target=True)
                state, reward, done, _ = test_env.step(action)
                total_reward += reward
                steps += 1
            print("Test env", total_reward, steps)
        print()
        if i % 10 == (episodes - 1) % 10:
            algo.save()
    print("Time:", time.time() - t)
    return algo


if __name__ == "__main__":
    is_test = True
    if is_test:
        t = time.time()
        real_env = make(ENV_NAME)
        state_dim, action_dim = real_env.observation_space.shape[0], real_env.action_space.shape[0]
        real_ddpg = train_agent(real_env, state_dim, action_dim, episodes=150)
        dataset = collect_trajectories(2048, real_ddpg, 0.2)
    else:
        dataset = load_dataset("dataset.csv")
        state_dim, action_dim = 3, 1
    worldmodel = train_worldmodel(dataset, state_dim, action_dim)
    fake_env = FakeEnv(worldmodel, [t[0][0] for t in dataset.trajectories], steps_limit=200)
    for _ in range(5):
        print()
    print("Fake env")
    fake_ddpg = train_agent(fake_env, state_dim, action_dim, test_env=(real_env if is_test else None), episodes=200)
