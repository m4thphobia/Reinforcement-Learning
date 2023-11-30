import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import torch.nn.functional as F

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, hidden_features = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dims, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
        )
        self.fc_actor = nn.Linear(hidden_features, n_actions)
        self.fc_critic = nn.Linear(hidden_features, 1)

    def forward(self, x):
        h = self.layers(x)
        action_prob = F.softmax(self.fc_actor(h), dim=-1)
        state_value = self.fc_critic(h)
        return action_prob, state_value

class ActorCriticAgent():
    def __init__(self, input_dims, n_actions, gamma = 0.99 ,lr=0.001) -> None:
        self.input_dims = input_dims
        self.gamma = gamma
        self.lr = lr
        self.memory = []
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.acnet = ActorCriticNetwork(input_dims, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.acnet.parameters(), lr=lr)
        self.memory = []

    def update_policy(self) -> None:
        G = 0
        actor_loss = 0
        critic_loss = 0
        for r, prob, v in self.memory[::-1]:
            G = r + self.gamma*G
            actor_loss -= (G-v) * torch.log(prob)
            critic_loss += F.smooth_l1_loss(v, torch.tensor(G).to(self.device))
        self.optimizer.zero_grad()
        loss = actor_loss + critic_loss
        loss.backward()
        self.optimizer.step()
        self.memory = []
    
    def get_action(self, state: np.ndarray) -> (int, float):
        state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.input_dims).to(self.device)
        action_prob, state_value = self.acnet(state_tensor.detach())
        action_prob, state_value = action_prob.squeeze(), state_value.squeeze()
        action = Categorical(action_prob).sample().item()
        return action, action_prob[action], state_value
    
    def get_greedy_action(self, state: np.ndarray) -> int:
        state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.input_dims).to(self.device)
        action_prob, _ = self.acnet(state_tensor.detach())
        action_prob = action_prob.squeeze()
        action = torch.argmax(action_prob.data).item()
        return action

    def add_memory(self, r: int, prob: float, v) -> None:
        self.memory.append((r, prob, v))