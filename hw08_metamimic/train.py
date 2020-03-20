from gym import make
import numpy as np
import torch
from torch.nn import functional as F
import copy
from collections import deque
import random
import time

GAMMA = 0.99
TAU = 0.01
HIDDEN_SIZE = 512
ENV_NAME = "Pendulum-v0"
device = torch.device("cuda")


def transform_state(state):
    return state


def collect_trajectories(count=2000, noise_std=0.025, train=True):
    env = make(ENV_NAME)
    trajectories = []
    agent = AgentImit()
    for _ in range(count):
        env.seed(42)
        state = env.reset()
        trajectory = [state]
        done = False
        while not done:
            if agent is not None:
                action = (agent.act(state) + noise_std * np.random.randn(env.action_space.shape[0])).clip(-1, 1)
            else:
                action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            trajectory.append(state)
        trajectories.append(trajectory)
        agent.reset()
    return trajectories


class AgentImit:
    def __init__(self):
        self.movement_type = 0
        self.noise_coef = 0.15
        self.noise = 0

    def reset(self):
        self.movement_type = (self.movement_type + 1) % 6
        self.noise = np.random.randn() * self.noise_coef

    def act(self, state):
        return np.clip(self._act(state) + self.noise, -1, 1)

    def _act(self, state):
        if self.movement_type == 0:
            return -1
        elif self.movement_type == 1:
            return -1 if state[0] < 0.5 else 0.8
        elif self.movement_type == 2:
            return (state[1] + state[0]) / 2
        elif self.movement_type == 3:
            return -np.sign(state[2]) if abs(state[2]) > 4 else (-1 if state[0] < 0 else 1)
        elif self.movement_type == 4:
            return -1 if state[1] < 0 else 1
        else:
            return -1 if state[0] < 0 else 1


class TD3:
    def __init__(self, state_dim, action_dim):
        self.memory = deque(maxlen=150000)
        self.gamma = GAMMA
        self.batch_size = 256
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(state_dim, HIDDEN_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_SIZE, action_dim),
            torch.nn.Tanh()
        )
        self.critic_one = torch.nn.Sequential(
            torch.nn.Linear(state_dim + action_dim, HIDDEN_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_SIZE, 1)
        )
        self.critic_two = torch.nn.Sequential(
            torch.nn.Linear(state_dim + action_dim, HIDDEN_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_SIZE, 1)
        )
        self.actor.to(device)
        self.critic_one.to(device)
        self.critic_two.to(device)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_one = copy.deepcopy(self.critic_one)
        self.target_critic_two = copy.deepcopy(self.critic_two)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=5e-4)
        self.critic_optim = torch.optim.Adam(list(self.critic_one.parameters()) + list(self.critic_two.parameters()), lr=5e-4)
        self.updates = 0

    def update(self, transition):
        self.memory.append(transition)
        if len(self.memory) < 2e3:
            return
        else:
            self.updates += 1
            state, action, next_state, reward, done = zip(*random.sample(self.memory, self.batch_size))
            state = torch.tensor(state, dtype=torch.float).to(device)
            action = torch.tensor(action, dtype=torch.float).to(device)
            next_state = torch.tensor(next_state, dtype=torch.float).to(device)
            reward = torch.tensor(reward, dtype=torch.float).to(device)

            #Critic
            with torch.no_grad():
                target_action = self.target_actor(next_state)
                target_action = (target_action + (0.05 * torch.randn(action.size(), device=device)).clamp_(-0.25, 0.25)).clamp_(-1, 1)
                q_true_1 = self.target_critic_one(torch.cat((next_state, target_action), dim=1))
                q_true_2 = self.target_critic_two(torch.cat((next_state, target_action), dim=1))
                q_true = torch.min(q_true_1, q_true_2)
                q_true[done] = 0
                q_true = reward + self.gamma * q_true.view(-1)
            q_pred = self.critic_one(torch.cat((state, action), dim=-1)).view(-1)
            q_pred_alt = self.critic_two(torch.cat((state, action), dim=-1)).view(-1)

            loss = ((q_true - q_pred) ** 2).mean() + ((q_true - q_pred_alt) ** 2).mean()
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()

            if self.updates % 2 == 0:
                for tp, sp in zip(self.target_critic_one.parameters(), self.critic_one.parameters()):
                    tp.data.copy_((1 - TAU) * tp.data + TAU * sp.data)
                for tp, sp in zip(self.target_critic_two.parameters(), self.critic_two.parameters()):
                    tp.data.copy_((1 - TAU) * tp.data + TAU * sp.data)

                #Actor

                loss = -(self.target_critic_one(torch.cat((state, self.actor(state)), dim=-1))).mean()
                self.actor_optim.zero_grad()
                loss.backward()
                self.actor_optim.step()

                for tp, sp in zip(self.target_actor.parameters(), self.actor.parameters()):
                    tp.data.copy_((1 - TAU) * tp.data + TAU * sp.data)

    def act(self, state, target=False):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
        with torch.no_grad():
            return self.actor(state)[0].cpu().numpy()

    def save(self):
        torch.save(self.target_actor, f"target_agent.pkl")
        torch.save(self.actor, f"agent.pkl")


def render(env, agent, trajectories):
    while True:
        reward_stage = trajectories[random.randint(0, len(trajectories) - 1)]
        done = False
        env.seed(42)
        state = env.reset()
        steps = 0
        state = np.concatenate((state, reward_stage[steps][0]))
        while not done:
            while not done:
                action = agent.act(state, target=True)
                state, reward, done, _ = env.step(action)
                state = np.concatenate((state, reward_stage[steps][2]))
                env.render()
                time.sleep(0.01)
                steps += 1


def train_agent(train_env, trajectories, episodes=250):
    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]
    agent = TD3(state_dim=state_dim * 2, action_dim=action_dim)
    eps = 0.1

    for i in range(episodes):
        train_env.seed(42)
        state = transform_state(train_env.reset())
        total_reward = 0
        steps = 0
        done = False
        trajectory = trajectories[i % len(trajectories)]
        state = np.concatenate((state, trajectory[1]))
        while not done:
            action = (agent.act(state) + np.random.randn(action_dim) * eps).clip(-1, 1)
            next_state, reward, done, _ = train_env.step(action)
            next_state = transform_state(next_state)
            reward = None  # TODO: Implement
            steps += 1
            next_state = np.concatenate((next_state, trajectory[steps]))
            agent.update((state, action, next_state, reward, done))
            state = next_state
            total_reward += reward

        print("Episode", i)
        print("Reward", total_reward)
        if i % 10 == (episodes - 1) % 10:
            agent.save()
    print("Time:", time.time() - t)
    return agent


if __name__ == "__main__":
    t = time.time()
    env = make(ENV_NAME)
    trajectories = collect_trajectories(128, noise_std=0.25)
    td3 = train_agent(env, trajectories=trajectories, episodes=300)
    render(env, td3, trajectories=collect_trajectories(128, train=False))
