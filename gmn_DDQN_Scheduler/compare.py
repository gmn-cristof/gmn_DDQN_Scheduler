import numpy as np
import matplotlib.pyplot as plt
from environment import KubernetesEnv
from traditional_schedulers.py import TraditionalSchedulers
from ddqn_agent import DDQNAgent

env = KubernetesEnv(num_nodes=5, num_pods=10)
schedulers = TraditionalSchedulers(env)

# Run traditional schedulers
random_rewards = []
round_robin_rewards = []
least_loaded_rewards = []

for _ in range(100):
    random_rewards.append(schedulers.random_scheduler())
    round_robin_rewards.append(schedulers.round_robin_scheduler())
    least_loaded_rewards.append(schedulers.least_loaded_scheduler())

# Run DDQN scheduler
agent = DDQNAgent(env.observation_space.shape, env.action_space.n)
agent.load("ddqn_model.h5")

ddqn_rewards = []

for _ in range(100):
    state = env.reset()
    state = np.reshape(state, [1, *env.observation_space.shape])
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, *env.observation_space.shape])
        total_reward += reward
        state = next_state

    ddqn_rewards.append(total_reward)

# Plot results
plt.figure(figsize=(12, 8))
plt.plot(random_rewards, label='Random Scheduler')
plt.plot(round_robin_rewards, label='Round Robin Scheduler')
plt.plot(least_loaded_rewards, label='Least Loaded Scheduler')
plt.plot(ddqn_rewards, label='DDQN Scheduler')
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Comparison of Scheduling Algorithms')
plt.show()
