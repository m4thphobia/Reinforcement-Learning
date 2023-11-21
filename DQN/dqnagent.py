import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
device = "cuda" if torch.cuda.is_available() else "cpu"


class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir) -> None:
        super().__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.conv1= nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2= nn.Conv2d(32, 64, 4, stride=2)
        self.conv3= nn.Conv2d(64, 64, 3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims,512)
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        state = torch.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv_state = conv3.view(conv3.size()[0], -1)
        flat1 = F.relu(self.fc1(conv_state))
        actions = self.fc2(flat1)

        return actions

    def save_checkpoint(self):
        print('...saving checkpointorch...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('...loading checkpointorch...')
        self.load_state_dict(torch.load(self.checkpoint_file))


##  append: dict({str:np.ndarray}) -> None
##  sample: batch_size: int -> memories: dict({str:[[np.ndarray] for batchsize]})
class ReplayBuffer():
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.memory = deque([], maxlen=memory_size)

    def append(self, transition: dict) -> None:
        self.memory.append(transition)

    def sample(self,batch_size: int) -> dict:
        indexes = np.random.randint(0, len(self.memory), batch_size)
        states = np.array([self.memory[index]["state"] for index in indexes])
        actions = np.array([self.memory[index]["action"] for index in indexes])
        rewards = np.array([self.memory[index]["reward"] for index in indexes])
        next_states = np.array([self.memory[index]["next_state"] for index in indexes])
        dones = np.array([self.memory[index]["done"] for index in indexes])
        terminateds = np.array([self.memory[index]["terminated"] for index in indexes])

        return {
            "states":states,
            "actions":actions,
            "rewards":rewards,
            "next_states":next_states,
            "dones":dones,
            "terminateds":terminateds
        }


class DQNAgent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                    mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                    replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.algo+'_q_eval',
                                    chkpt_dir=self.chkpt_dir)

        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.algo+'_q_next',
                                    chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation],dtype=torch.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, reward, action, state_, done, terminated):
        transition = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": state_,
            "done": int(done),
            "terminated": int(terminated),
        }
        self.memory.append(transition)

    def sample_memory(self):
        memories = self.memory.sample(self.batch_size)

        states = torch.tensor(memories["state"]).to(self.q_eval.device)
        actions = torch.tensor(memories["action"]).to(self.q_eval.device)
        rewards = torch.tensor(memories["reward"]).to(self.q_eval.device)
        states_ = torch.tensor(memories["next_state"]).to(self.q_eval.device)
        dones = torch.tensor(memories["done"]).to(self.q_eval.device)
        terminateds = torch.tensor(memories["terminated"]).to(self.q_eval.device)

        return states, actions, rewards, states_, dones, terminateds

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones, terminated = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
