from gym import make
from agent import Agent


if __name__ == "__main__":
    env = make("MountainCar-v0")
    algo = Agent(state_dim=2, action_dim=3)
    episodes = 100
    visit_count = 0

    for i in range(episodes):
        state = env.reset()
        steps = 0
        done = False
        while not done:
            action = dqn.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state
            steps += 1
            algo.update((state, action, next_state, reward, done))
            state = next_state
        if steps < 200:
            visit_count += 1
            print("Visited target state at episode", i)
    print()
    print("Total visit count:", visit_count)