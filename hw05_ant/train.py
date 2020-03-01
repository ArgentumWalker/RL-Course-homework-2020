import pubullet_envs
from gym import make
import numpy as np
import torch
import random
import copy

N_STEP = 1
GAMMA = 0.99
TAU = 0.001
HIDDEN_SIZE = 256
CRITIC_LR = 5e-4
ACTOR_LR = 2e-4


def transform_state(state):
    return np.array(state)


def softupdate(target, source):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - TAU) * tp.data + TAU * sp.data)


class TD3:
    def __init__(self, state_dim, action_dim):
        self.gamma = GAMMA ** N_STEP
        self.actor = None # Torch model
        self.target_actor = copy.deepcopy(self.actor)
        self.critic = None # Torch model
        self.critic_alt = None # Torch model
        self.target_critic = copy.deepcopy(self.critic)
        self.target_critic_alt = copy.deepcopy(self.critic_alt)

    def update(self, transition):
        state, action, next_state, reward, done = transition
        soft_update(self.target_critic, self.critic)
        soft_update(self.target_critic_alt, self.critic_alt)
        soft_update(self.target_actor, self.actor)

    def act(self, state):
        return None

    def save(self, path):
        torch.save(self.actor, "agent.pkl")


def exploit(agent, env, episodes=10):
    total_reward = 0
    for _ in range(episodes):
        state = transform_state(env.reset())
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = transform_state(next_state)
            total_reward += reward
            state = next_state
    return total_reward / episodes


if __name__ == "__main__":
    env = make("AntBulletEnv-v0")
    algo = TD3(state_dim=3, action_dim=1)
    episodes = 1000
    best_reward = -1e6

    for i in range(episodes):
        state = transform_state(env.reset())
        total_reward = 0
        steps = 0
        done = False
        reward_buffer = deque(maxlen=N_STEP)
        state_buffer = deque(maxlen=N_STEP)
        action_buffer = deque(maxlen=N_STEP)
        while not done:
            action = dqn.act(state)
            # Use OUNoise or NormalNoise here
            next_state, reward, done, _ = env.step(action)
            next_state = transform_state(next_state)
            total_reward += reward
            steps += 1
            reward_buffer.append(reward)
            state_buffer.append(state)
            action_buffer.append(action)
            if len(reward_buffer) == N_STEP:
                algo.update((state_buffer[0], action_buffer[0], next_state, sum([(GAMMA ** i) * r for i, r in enumerate(reward_buffer)]), done))
            state = next_state
        if len(reward_buffer) == N_STEP:
            rb = list(reward_buffer)
            for k in range(1, N_STEP):
                algo.update((state_buffer[k], action_buffer[k], next_state, sum([(GAMMA ** i) * r for i, r in enumerate(rb[k:])]), done))

        if i % 20 == 0:
            exploit_reward = exploit(algo, env)
            if best_reward < exploit_reward:
                best_reward = exploit_reward
                algo.save()
