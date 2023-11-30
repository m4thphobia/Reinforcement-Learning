import numpy as np
import torch
import torch.nn as nn
import toch.optim as optim
from torch.distributions import Categorical, Normal

class PolicyNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, hidden_features = 128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dims, hidden_features),
            nn.ELU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ELU(),
            nn.Linear(hidden_features, n_actions),
            nn.Softmax(dim=-1)
        )
        self.

    def forward(self, x):
        return self.layers(x)

class PolicyGradientAgen():
    def __init__(self, input_dims, n_actions, gamma = 0.99 ,lr=0.001) -> None:
        self.input_dims = input_dims
        self.gamma = gamma
        self.lr = lr
        self.memory = []
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.pi = PolicyNetwork(input_dims, n_actions).to(device)
        self.optimizer = optim.Adam(self.pi.parameters(), lr=lr)

    def update_policy(self) -> None:
        G = 0
        loss = 0
        for r, prob in self.memory[::-1]:
            G = r + self.gamma*G
            loss -= G * torch.log(prob) 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def get_action(self, state: np.ndarray) -> (int, float):
        state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.input_dims).to(self.device)
        action_probs = self.pi(state_tensor).squeeze()
        print(f'action_probs: {action_probs} |device: {action_probs.device}') #!
        action = Categorical(action_prob).sample().item()
        print(f'action: {action} |device: {action.device}') #!
        self.memory.append((action, action_prob[action]))
        return action
    
    def get_greedy_action(self, state: np.ndarray) -> int:
        state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.input_dims).to(self.device)
        action_prob = self.pinet(state_tensor.data).squeeze()
        action = torch.argmax(action_prob.data).item()
        print(f'action: {action} |device: {action.device}') #!
        return action

    def add_memory(self, r: int, prob: float) -> None:
        self.memory.append((r, prob))

    def reset_memory(self) -> None:
        self.memory = []