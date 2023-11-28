import os
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import matplotlib.pyplot as plt
from matplotlib import animation


class PolicyNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int=128) -> None:
        super().__init__()
        self.f1 = nn.Linear(state_size, hidden_size)
        self.f2 = nn.Linear(hidden_size, hidden_size)
        self.f3 = nn.Linear(hidden_size, action_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.f1(state))
        h = F.elu(self.f2(h))
        action_prob = F.softmax(self.f3(h), dim=-1)
        return action_prob


class ReinforceAgent:
    def __init__(self, state_size: int, action_size: int, alpha=0.99, lr=0.0005) -> None:
        self.state_size = state_size
        self.policy_net = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.alpha = torch.tensor(alpha).to(self.policy_net.device)
        self.memory=[]

    def get_greedy_action(self, state: np.ndarray) -> int:
        state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.state_size).to(self.policy_net.device)
        action_probs = self.policy_net(state_tensor.detach()).squeeze()
        action = torch.argmax(action_probs).item()
        return action

    def get_action(self, state: np.ndarray) -> (int, torch.Tensor):
        state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.state_size).to(self.policy_net.device)
        action_probs = self.policy_net(state_tensor.detach()).squeeze()
        action = Categorical(action_probs).sample().item()
        return action, action_probs[action]

    def add_memory(self, reward: int, prob: float) -> None:
        self.memory.append((reward, prob))

    def update(self):
        G = 0
        loss = 0
        for reward, prob in reversed(self.memory):
            G = reward + self.alpha*G
            loss -= G*torch.log(prob)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory = []


if __name__ == '__main__':
    env = gym.make('LunarLander-v2', render_mode="rgb_array")
    agent = ReinforceAgent(env.observation_space.shape[0], env.action_space.n)

    episodes = 300
    max_steps = env.spec.max_episode_steps
    episode_returns = []

    for episode in range(episodes):
        state, _ = env.reset()
        episode_return = 0
        for _ in range(max_steps):
            action, prob = agent.get_action(state)
            next_state, reward, done, terminated, _ = env.step(action)
            episode_return += reward

            reward = torch.tensor(reward,device=agent.policy_net.device)
            agent.add_memory(reward, prob)
            if done or terminated:
                agent.update()
                break

            state=next_state

        episode_returns.append(episode_return)
        if episode % 20 == 0:
            print("Episode %d finished | Episode reward %f" % (episode, episode_return))
