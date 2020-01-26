from gym import make
import numpy as np
import torch
import random

N_STEP = 1
GAMMA = 0.9


def transform_state(state):
    return np.array(state)


class DQN:
    def __init__(self, state_dim, action_dim):
        self.gamma = GAMMA ** N_STEP
        self.model = None # Torch model

    def update(self, transition):
        state, action, next_state, reward, done = transition

    def act(self, state, target=False):
        return 0

    def save(self, path):
        torch.save(self.model, "agent.pkl")


if __name__ == "__main__":
    env = make("LunarLander-v2")
    dqn = DQN(state_dim=8, action_dim=4)
    eps = 0.1
    episodes = 1000

    for i in range(episodes):
        state = transform_state(env.reset())
        total_reward = 0
        steps = 0
        done = False
        reward_buffer = deque(maxlen=N_STEP)
        state_buffer = deque(maxlen=N_STEP)
        action_buffer = deque(maxlen=N_STEP)
        while not done:
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = dqn.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = transform_state(next_state)
            total_reward += reward
            steps += 1
            reward_buffer.append(reward)
            state_buffer.append(state)
            action_buffer.append(action)
            if len(reward_buffer) == N_STEP:
                dqn.update((state_buffer[0], action_buffer[0], next_state, sum([(GAMMA ** i) * r for i, r in enumerate(reward_buffer)]), done))
            state = next_state
        if len(reward_buffer) == N_STEP:
            rb = list(reward_buffer)
            for k in range(1, N_STEP):
                dqn.update((state_buffer[k], action_buffer[k], next_state, sum([(GAMMA ** i) * r for i, r in enumerate(rb[k:])]), done))

        if i % 20 == 0:
            dqn.save()
