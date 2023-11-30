import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict, deque
import copy


class QNetwork(nn.Module):
    def __init__(self, input_dims: np.ndarray, n_actions: int, lr: int=0.01 ) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_dims[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )

        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.linear_block = nn.Sequential(
            nn.Linear(fc_input_dims, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        state = torch.zeros(1, *input_dims)
        dims = self.conv_block(state)
        return int(np.prod(dims.size()))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        conv = self.conv_block(state)
        conv_state = conv.view(conv.size()[0], -1)
        actions = self.linear_block(conv_state)

        return actions


class ReplayBuffer:
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.memory = deque([], maxlen=memory_size)

    def append(self, transition: dict) -> None:
        self.memory.append(transition)

    def sample(self, batch_size: int) -> dict:
        batch_indexes = np.random.randint(0, len(self.memory), size=batch_size)
        states = np.array([self.memory[index]["state"] for index in batch_indexes])
        next_states = np.array([self.memory[index]["next_state"] for index in batch_indexes])
        rewards = np.array([self.memory[index]["reward"] for index in batch_indexes])
        actions = np.array([self.memory[index]["action"] for index in batch_indexes])
        dones = np.array([self.memory[index]["done"] for index in batch_indexes])
        return {
            "states": states,
            "next_states": next_states,
            "rewards": rewards,
            "actions": actions,
            "dones": dones,
        }


class DqnAgent:
    def __init__(self, input_dims, n_actions, gamma=0.99, epsilon=1.0, lr=0.001,
                    mem_size=50000, batch_size=32, eps_min=0.01, eps_dec=5e-7,
                    replace=1000):

        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace

        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        self.replay_buffer = ReplayBuffer(mem_size)

        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.qnet = QNetwork(self.input_dims, self.n_actions, self.lr).to(self.device)
        self.target_qnet = QNetwork(self.input_dims, self.n_actions, self.lr).to(self.device)
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)


        self.learn_step_counter = 0
        self.replace_target_cnt = 500

    def update_q(self) -> None:

        self.replace_target_network()

        batch = self.sample_memory()
        q = self.qnet(batch["states"])
        targetq = copy.deepcopy(q.detach().to("cpu").numpy())


        maxq = torch.max(self.target_qnet(batch["next_states"]), dim=1).values.detach().to("cpu").numpy()

        for i in range(self.batch_size):
            targetq[i, batch["actions"][i]] = batch["rewards"][i] + self.gamma*maxq[i]*(not batch["dones"][i])

        self.optimizer.zero_grad()
        loss = nn.MSELoss()(q, torch.tensor(targetq).to(self.device))
        loss.backward()
        self.optimizer.step()
        self.learn_step_counter += 1

    def get_greedy_action(self, state: np.ndarray) -> int:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action = torch.argmax(self.qnet(state_tensor).detach()).item()
        return action

    def get_action(self, state: np.ndarray) -> int:
        self.decrement_epsilon()
        if self.epsilon <= np.random.uniform(0, 1):
            action = self.get_greedy_action(state)
        else:
            action = np.random.choice(self.action_space)
        return action

    def store_transition(self, state, reward, action, next_state, done):
        transition = {
            "state": state,
            "reward": reward,
            "action": action,
            "next_state": next_state,
            "done": int(done),
        }
        self.replay_buffer.append(transition)

    def sample_memory(self):
        sampled_memories = self.replay_buffer.sample(self.batch_size)
        sampled_memories["states"] =  torch.tensor(sampled_memories["states"], dtype=torch.float32).to(self.device)
        sampled_memories["next_states"] =  torch.tensor(sampled_memories["next_states"], dtype=torch.float32).to(self.device)
        return  sampled_memories

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.target_qnet.load_state_dict(self.qnet.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min


# class DQNAgent():
#     def __init__(self, input_dims, n_actions, gamma=0.99, epsilon=1.0, lr=0.001,
#                     mem_size=50000, batch_size=32, eps_min=0.01, eps_dec=5e-7,
#                     replace=1000):

#         self.input_dims = input_dims
#         self.n_actions = n_actions
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.lr = lr
#         self.batch_size = batch_size
#         self.eps_min = eps_min
#         self.eps_dec = eps_dec
#         self.replace_target_cnt = replace
#         #self.algo = algo
#         #self.env_name = env_name
#         #self.chkpt_dir = chkpt_dir
#         self.action_space = [i for i in range(n_actions)]
#         self.learn_step_counter = 0

#         self.memory = ReplayBuffer(mem_size)

#         self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
#         self.qnet = QNetwork(self.input_dims, self.n_actions, self.lr)
#         self.target_qnet = QNetwork(self.input_dims, self.n_actions, self.lr)

#     def get_action(self, state: np.ndarray) -> int:
#         if np.random.random() > self.epsilon:
#             state = torch.tensor(state, dtype=torch.float).view(-1, ).to(self.device)
#             actions = self.q_eval.forward(state)
#             action = torch.argmax(actions).item()
#         else:
#             action = np.random.choice(self.action_space)

#         return action

#     def get_greeedy_action(sel, state: np.ndarray) -> int:
#         pass

#     def store_transition(self, state, reward, action, state_, done, terminated):
#         transition = {
#             "state": state,
#             "action": action,
#             "reward": reward,
#             "next_state": state_,
#             "done": int(done),
#             "terminated": int(terminated),
#         }
#         self.memory.append(transition)

#     def sample_memory(self):
#         memories = self.memory.sample(self.batch_size)

#         states = torch.tensor(memories["state"]).to(self.q_eval.device)
#         actions = torch.tensor(memories["action"]).to(self.q_eval.device)
#         rewards = torch.tensor(memories["reward"]).to(self.q_eval.device)
#         states_ = torch.tensor(memories["next_state"]).to(self.q_eval.device)
#         dones = torch.tensor(memories["done"]).to(self.q_eval.device)
#         terminateds = torch.tensor(memories["terminated"]).to(self.q_eval.device)

#         return states, actions, rewards, states_, dones, terminateds

#     def replace_target_network(self):
#         if self.learn_step_counter % self.replace_target_cnt == 0:
#             self.q_next.load_state_dict(self.q_eval.state_dict())

#     def decrement_epsilon(self):
#         self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

#     def save_models(self):
#         self.q_eval.save_checkpoint()
#         self.q_next.save_checkpoint()

#     def load_models(self):
#         self.q_eval.load_checkpoint()
#         self.q_next.load_checkpoint()

#     def learn(self):
#         if self.memory.mem_cntr < self.batch_size:
#             return

#         self.q_eval.optimizer.zero_grad()

#         self.replace_target_network()

#         states, actions, rewards, states_, dones, terminated = self.sample_memory()
#         indices = np.arange(self.batch_size)

#         q_pred = self.q_eval.forward(states)[indices, actions]
#         q_next = self.q_next.forward(states_).max(dim=1)[0]

#         q_next[dones] = 0.0
#         q_target = rewards + self.gamma*q_next

#         loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
#         loss.backward()
#         self.q_eval.optimizer.step()
#         self.learn_step_counter += 1

#         self.decrement_epsilon()

