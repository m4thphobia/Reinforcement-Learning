import numpy as np
import torch
from torch import nn
import torch.optim as optim
from collections import defaultdict, deque
import copy


class QNetwork(nn.Module):
    def __init__(self, num_state: int, num_action: int, hidden_size: int = 16) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(num_state, hidden_size),
                                    nn.ELU(),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.ELU(),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.ELU(),
                                    nn.Linear(hidden_size, num_action),
                                    nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ReplayBuffer:
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.memory = deque([], maxlen=memory_size)

    def append(self, transition: dict) -> None:
        self.memory.append(transition)

    def sample(self, batch_size: int) -> dict:
        batch_indexes = np.random.randint(0, len(self.memory), size=batch_size)
        states = np.array([self.memory[index]["state"] for index in batch_indexes])
        rewards = np.array([self.memory[index]["reward"] for index in batch_indexes])
        actions = np.array([self.memory[index]["action"] for index in batch_indexes])
        next_states = np.array([self.memory[index]["next_state"] for index in batch_indexes])
        dones = np.array([self.memory[index]["done"] for index in batch_indexes])
        return {
            "states": states,
            "rewards": rewards,
            "actions": actions,
            "next_states": next_states,
            "dones": dones,
        }

class DqnAgent:
    def __init__(
        self,
        num_state: int,
        num_action: int,
        gamma: float = 0.99,
        lr: float = 0.001,
        batch_size: int = 32,
        memory_size: int = 50000,
    ) -> None:
        self.device = "cpu" #"cuda:1" if torch.cuda.is_available() else "cpu"
        self.num_state = num_state
        self.num_action = num_action
        self.gamma = gamma
        self.batch_size = batch_size
        self.qnet = QNetwork(num_state, num_action).to(self.device)
        self.target_qnet = QNetwork(num_state, num_action).to(self.device)
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(memory_size)

        self.learn_step_counter = 0
        self.replace_target_cnt = 500

    def update_q(self) -> None:
        batch = self.replay_buffer.sample(self.batch_size)
        q = self.qnet(torch.tensor(batch["states"], dtype=torch.float).to(self.device))
        targetq = copy.deepcopy(q.detach().to("cpu").numpy())

        maxq = torch.max(
            self.target_qnet(torch.tensor(batch["next_states"], dtype=torch.float).to(self.device)),
            dim=1,
        ).values

        for i in range(self.batch_size):
            targetq[i, batch["actions"][i]] = batch["rewards"][i] + self.gamma*maxq[i]*(not batch["dones"][i])
        self.optimizer.zero_grad()
        loss = nn.MSELoss()(q, torch.tensor(targetq).to(self.device))
        loss.backward()
        self.optimizer.step()
        self.learn_step_counter += 1
        self.replace_target_network()

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.target_qnet.load_state_dict(self.qnet.state_dict())

    def get_greedy_action(self, state: np.ndarray) -> int:
        state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.num_state).to(self.device)
        action = torch.argmax(self.qnet(state_tensor).detach()).item()
        return action

    def get_action(self, state: np.ndarray, episode: int) -> int:
        epsilon = 0.7 * (1 / (episode + 1))
        if epsilon <= np.random.uniform(0, 1):
            action = self.get_greedy_action(state)
        else:
            action = np.random.choice(self.num_action)
        return action
