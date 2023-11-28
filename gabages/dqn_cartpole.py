import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import animation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import copy
device = "cuda" if torch.cuda.is_available() else "cpu"
def fix_seed(seed: int=1) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
fix_seed()
##  forward: state: torch.tensor -> action_probs: torch.tensor
class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 16) -> None:
        super().__init__()

        self.fc1= nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.fc1(x))
        h = F.elu(self.fc2(h))
        h = F.elu(self.fc3(h))
        y = F.elu(self.fc4(h))
        return y

def QNetwork_exam():
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    state_size = env.observation_space.shape[0]
    action_size =  env.action_space.n

    qnet = QNetwork(state_size, action_size).to(device)

    state = torch.rand(env.observation_space.shape[0], device=device)
    print(f"state:{state}")
    action = qnet.forward(state)
    print(f"action:{action}")

##  append: dict({str:np.ndarray}) -> None
##  sample: batch_size: int -> memories: dict({str:[[torch.tensor] for batchsize]})
class ReplayBuffer():
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.memory = deque([], maxlen=memory_size)

    def append(self, transition: dict) -> None:
        self.memory.append(transition)

    def sample(self,batch_size: int) -> dict:
        indexes = np.random.randint(0, len(self.memory), batch_size)
        states = torch.tensor([self.memory[index]["state"] for index in indexes])
        next_states = torch.tensor([self.memory[index]["next_state"] for index in indexes])
        actions = torch.tensor([self.memory[index]["action"] for index in indexes])
        rewards = torch.tensor([self.memory[index]["reward"] for index in indexes])
        dones = torch.tensor([self.memory[index]["done"] for index in indexes])
        terminateds = torch.tensor([self.memory[index]["terminated"] for index in indexes])

        return {
            "states":states,
            "next_states":next_states,
            "actions":actions,
            "rewards":rewards,
            "dones":dones,
            "terminateds":terminateds
        }

def ReplayBuffer_exam():
    memory_size = 50000
    batch_size = 32
    initial_memory_size = 500
    replay_buffer = ReplayBuffer(memory_size)

    env = gym.make('CartPole-v1', render_mode="rgb_array")
    state_size = env.observation_space.shape[0]
    action_size =  env.action_space.n

    for _ in range(initial_memory_size):
        action = np.random.rand(action_size)
        state = np.random.rand(state_size)
        next_state = np.random.rand(state_size)
        reward = np.random.rand(1)
        done = np.random.choice([True, False])
        terminated = np.random.choice([True, False])

        transition = {
            "state": state,
            "next_state": next_state,
            "reward": reward,
            "action": action,
            "done": int(done),
            "terminated": int(terminated),
        }
        replay_buffer.append(transition)
        if done or terminated:
            state, _ = env.reset()
        else:
            state = next_state

    batch = replay_buffer.sample(batch_size)
    print(batch["actions"])
    print(f"memory length: {len(replay_buffer.memory)}")

##  get_action: state: np.ndarray -> action: int
##  update:
class DqnAgent():
    def __init__(self,state_size: int, action_size: int, gamma: float = 0.99, epsilon: float = 0.7, lr: float = 0.001,
        batch_size: int = 32, memory_size: int = 50000) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size

        self.qnet = QNetwork(state_size, action_size).to(device)
        self.target_qnet = copy.deepcopy(self.qnet)
        self.replay_buffer = ReplayBuffer(memory_size)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)

    def get_greedy_action(self, state: np.ndarray) -> int:
        state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.state_size).to(device)
        action = torch.argmax(self.qnet(state_tensor).detach()).item()
        return action

    def get_action(self, state: np.ndarray, episode: int) -> int:
        epsilon = self.epsilon/(1+episode)
        if epsilon <= np.random.uniform(0, 1):
            action = self.get_greedy_action(state)
        else:
            action = np.random.choice(self.action_size)
        return action

    def update_q(self):
        batch = self.replay_buffer.sample(self.batch_size)
        q = self.qnet(batch["states"].to(device))
        # ×targetq = self.target_qnet(torch.tensor(batch["states"], dtype=torch.float))
        targetq = copy.deepcopy(q.detach())
        maxq = torch.max(
            self.target_qnet(batch["next_states"].to(device)),
            dim=1,
        ).values
        for i in range(self.batch_size):
            targetq[i][batch["actions"][i]] = batch["rewards"][i] + self.gamma*maxq[i]*(not batch["dones"][i])
        self.optimizer.zero_grad()
        loss = nn.MSELoss()(q, targetq)
        loss.backward()
        self.optimizer.step()

    def sync_qnet(self):
        self.target_qnet = copy.deepcopy(self.qnet)

def DqnAgent_exam():
    env = gym.make('CartPole-v1', render_mode="rgb_array") ## cpu
    agent = DqnAgent(env.observation_space.shape[0], env.action_space.n)
    initial_memory_size = 500

    state, _ = env.reset() ## cpu
    for step in range(initial_memory_size):
        action = env.action_space.sample()
        print(action.device)
        next_state, reward, done, terminated, _ = env.step(action)
        transition = {
            "state": state,
            "next_state": next_state,
            "reward": reward,
            "action": action,
            "done": int(done),
            "terminated": int(terminated),
        }
        agent.replay_buffer.append(transition)
        if done or terminated:
            state, _ = env.reset()
        else:
            state = next_state

    max_steps = env.spec.max_episode_steps
    state, _ = env.reset()
    episode_reward = 0
    for t in range(max_steps):
        action = agent.get_action(state, 0)  # 行動を選択
        next_state, reward, done, terminated, _ = env.step(action)
        episode_reward += reward
        transition = {
            "state": state,
            "next_state": next_state,
            "reward": reward,
            "action": action,
            "done": int(done),
            "terminated": int(terminated),
        }
        agent.replay_buffer.append(transition)
        agent.update_q()  # Q関数を更新
        if done or terminated:
            state, _ = env.reset()
        else:
            state = next_state
    print(episode_reward)

#ReplayBuffer_exam()
num_episode = 2000
memory_size = 50000
initial_memory_size = 500


episode_rewards = []
num_average_epidodes = 10

env = gym.make('CartPole-v1', render_mode="rgb_array")
max_steps = env.spec.max_episode_steps

agent = DqnAgent(env.observation_space.shape[0], env.action_space.n)

##  stock some memory
state, _ = env.reset()
for step in range(initial_memory_size):
    action = env.action_space.sample()  # ランダムに行動を選択
    next_state, reward, done, terminated, _ = env.step(action)
    transition = {
        "state": state,
        "next_state": next_state,
        "reward": reward,
        "action": action,
        "done": int(done),
        "terminated": int(terminated),
    }
    agent.replay_buffer.append(transition)
    if done or terminated:
        state, _ = env.reset()
    else:
        state = next_state

##  Training loop
for episode in range(num_episode):
    state, _ = env.reset()
    episode_reward = 0
    for t in range(max_steps):
        action = agent.get_action(state, episode=episode)
        next_state, reward, done, terminated, _ = env.step(action)
        episode_reward += reward
        transition = {
            "state": state,
            "next_state": next_state,
            "reward": reward,
            "action": action,
            "done": int(done),
            "terminated": int(terminated),
        }
        agent.replay_buffer.append(transition)
        agent.update_q()
        state = next_state
        if done or terminated:
            break

    episode_rewards.append(episode_reward)
    if episode % 20 == 0:
        agent.sync_qnet()
        print(f"Episode {episode} finished | Episode reward {episode_reward}")


if not os.path.exists("../out"):
    os.makedirs("../out")

##  save model
torch.save(obj=agent.qnet.state_dict(), f=f"../out/{os.path.basename(__file__).split('.')[0]}.pth")

## moving average
moving_average = np.convolve(
    episode_rewards, np.ones(num_average_epidodes) / num_average_epidodes, mode="valid"
)

# plot
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_title(f"{os.path.basename(__file__).split('.')[0]}")
ax.plot(np.arange(len(moving_average)), moving_average)
ax.set_xlabel("episode")
ax.set_ylabel("rewards")
plt.savefig(f"../out/{os.path.basename(__file__).split('.')[0]}.png")

##  save gif
def save_frames_as_gif(frames, path="../out", filename=f'/{os.path.basename(__file__).split(".")[0]}.gif'):

    if not os.path.exists(path):
        os.makedirs(path)

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

##  Testing loop
frames = []
episodes = 3
for episode in range(episodes):
    state, _ = env.reset()
    frames.append(env.render())
    for t in range(max_steps):
        action = agent.get_greedy_action(state)
        next_observation, reward, done, terminated, _ = env.step(action)
        rendered_frame = env.render()
        if rendered_frame is not None:
            frames.append(rendered_frame)
        if done or terminated:
            break

env.close()
save_frames_as_gif(frames)
