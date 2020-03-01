from gym import make
import numpy as np
import torch
import random

GAMMA = 0.9
CLIP = 0.1
ENTROPY_COEF = 1e-2
TRAJECTORY_SIZE = 512


def transform_state(state):
    return np.array(state)


class PPO:
    def __init__(self, state_dim, action_dim):
        self.gamma = GAMMA ** N_STEP
        self.actor = None # Torch model
        self.critic = None # Torch model

    def update(self, trajectory):
        state, action, rollouted_reward = zip(*transition)
        
    def get_value(self, state):
        # Should return expected value of the state
        return 0

    def act(self, state):
        # Remember: agent is not deterministic, sample actions from distribution (e.g. Gaussian)
        return 0

    def save(self, path):
        torch.save(self.actor, "agent.pkl")


if __name__ == "__main__":
    env = make("HalfCheetahBulletEnv-v0")
    algo = PPO(state_dim=26, action_dim=6)
    episodes = 10000

    reward_buffer = deque()
    state_buffer = deque()
    action_buffer = deque()
    done_buffer = deque()
    for i in range(episodes):
        state = transform_state(env.reset())
        total_reward = 0
        steps = 0
        done = False
        while not done:
            action = dqn.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = transform_state(next_state)
            total_reward += reward
            steps += 1
            reward_buffer.append(reward)
            state_buffer.append(state)
            action_buffer.append(action)
            if len(action_buffer) == TRAJECTORY_SIZE:
                rollouted_reward = [algo.get_value(state) if not done else 0]
                for r, d in zip(reward_buffer, done_buffer):
                    rollouted_reward.append(r + GAMMA * d * rb[-1])
                rollouted_reward = list(reversed(rollouted_reward))
                trajectory = []
                for k in range(0, len(state_buffer)):
                    trajectory.append((state_buffer[k], action_buffer[k], rollouted_reward[k]))
                algo.update(trajectory)
                action_buffer.clear()
                reward_buffer.clear()
                state_buffer.clear()
                done_buffer.clear()

        if i % 50 == 0:
            algo.save()
