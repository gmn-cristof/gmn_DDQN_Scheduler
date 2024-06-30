import gym
from gym import spaces
import numpy as np

class KubernetesEnv(gym.Env):
    def __init__(self, num_nodes=5, num_pods=10):
        super(KubernetesEnv, self).__init__()

        self.action_space = spaces.Discrete(num_nodes)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(num_nodes, 7), dtype=np.float32)

        self.num_nodes = num_nodes
        self.num_pods = num_pods
        self.nodes = np.random.rand(num_nodes, 7)
        self.pods = self.generate_pods(num_pods)

        self.current_pod = 0

    def generate_pods(self, num_pods):
        pods = []
        for _ in range(num_pods):
            pod_type = np.random.choice(['cpu', 'gpu', 'io', 'network', 'average'])
            if pod_type == 'cpu':
                pods.append(np.random.rand(7) * [0.7, 0, 0.5, 0.5, 0.5, 0.5, 0.5])
            elif pod_type == 'gpu':
                pods.append(np.random.rand(7) * [0.5, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5])
            elif pod_type == 'io':
                pods.append(np.random.rand(7) * [0.5, 0.5, 0.5, 0.7, 0.5, 0.5, 0.5])
            elif pod_type == 'network':
                pods.append(np.random.rand(7) * [0.5, 0.5, 0.5, 0.5, 0.7, 0.5, 0.5])
            elif pod_type == 'average':
                pods.append(np.random.rand(7) * [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        return np.array(pods)

    def reset(self):
        self.nodes = np.random.rand(self.num_nodes, 7)
        self.pods = self.generate_pods(self.num_pods)
        self.current_pod = 0
        return self.nodes

    def step(self, action):
        if action < 0 or action >= self.num_nodes:
            raise ValueError(f"Action {action} is out of bounds for {self.num_nodes} nodes")
        
        pod = self.pods[self.current_pod]
        self.nodes[action] += pod

        reward = self.compute_reward(action, pod)

        self.current_pod += 1
        done = self.current_pod >= self.num_pods

        return self.nodes, reward, done, {}

    def compute_reward(self, action, pod):
        # 超参数和权重
        alpha = 0.5  # 减小alpha的值
        beta = 2.0   # 增大beta的值
        gamma = 2.0  # 增大gamma的值
        weights = np.array([1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # 调整资源权重

        # 理想状态差异惩罚
        ideal_state = 0.5
        load_diff = np.sum(weights * (self.nodes - ideal_state) ** 2, axis=1)
        reward = -alpha * np.sum(load_diff)

        # 过载惩罚
        overload_penalty = np.sum(weights * np.maximum(self.nodes - 1.0, 0), axis=1)
        reward -= beta * np.sum(overload_penalty)

        # 匹配奖励
        match_reward = np.sum(weights * pod * self.nodes[action])
        reward += gamma * match_reward

        return reward

    def render(self, mode='human'):
        pass
