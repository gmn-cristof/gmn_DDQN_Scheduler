import numpy as np

class TraditionalSchedulers:
    def __init__(self, env):
        self.env = env

    def random_scheduler(self):
        env = self.env
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            total_reward += reward

        return total_reward

    def round_robin_scheduler(self):
        env = self.env
        state = env.reset()
        total_reward = 0
        done = False
        current_node = 0

        while not done:
            action = current_node
            state, reward, done, _ = env.step(action)
            total_reward += reward
            current_node = (current_node + 1) % env.num_nodes

        return total_reward

    def least_loaded_scheduler(self):
        env = self.env
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            load = np.sum(state, axis=1)
            action = np.argmin(load)
            state, reward, done, _ = env.step(action)
            total_reward += reward

        return total_reward
