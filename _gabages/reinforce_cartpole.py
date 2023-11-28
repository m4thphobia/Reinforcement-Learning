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
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = F.elu(self.fc2(h))
        action_prob = F.softmax(self.fc3(h), dim=-1)
        return action_prob

class ReinforceAgent():
    def __init__(self, state_size: int, action_size: int, gamma=0.99, lr=0.001) -> None:
        self.state_size = state_size
        self.gamma = gamma
        self.pinet = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.pinet.parameters(), lr=lr)
        self.memory = []

    def get_greedy_action(self, state: np.ndarray) -> int:
        state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.state_size)
        action_prob = self.pinet(state_tensor.data).squeeze()
        action = torch.argmax(action_prob.data).item()
        return action

    def get_action(self, state: np.ndarray) -> (int, float):
        state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.state_size)
        action_prob = self.pinet(state_tensor.detach()).squeeze()
        action = Categorical(action_prob).sample().item()
        return action, action_prob[action]

    def add_memory(self, reward: int, prob: float) -> None:
        self.memory.append((reward, prob))

    def update(self) -> None:
        R = 0
        loss = 0
        for reward, prob in reversed(self.memory):
            R = reward + self.gamma*R
            loss -= R*torch.log(prob)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory = []


num_episode = 600  # 学習エピソード数

# ログ
episode_rewards = []
num_average_epidodes = 10

env = gym.make("CartPole-v1", render_mode="rgb_array")
max_steps = env.spec.max_episode_steps  # エピソードの最大ステップ数

agent = ReinforceAgent(env.observation_space.shape[0], env.action_space.n)

for episode in range(num_episode):
    state, _ = env.reset()  # envからは4次元の連続値の観測が返ってくる
    episode_reward = 0
    for t in range(max_steps):
        action, prob = agent.get_action(state)  #  行動を選択
        next_state, reward, done, terminated, _ = env.step(action)
        episode_reward += reward
        agent.add_memory(reward, prob)
        state = next_state
        # 　エピソードが終了したら
        if done or terminated:
            agent.update()
            break
    episode_rewards.append(episode_reward)
    if episode % 20 == 0:
        print("Episode %d finished | Episode reward %f" % (episode, episode_reward))

# 累積報酬の移動平均を表示
moving_average = np.convolve(
    episode_rewards, np.ones(num_average_epidodes) / num_average_epidodes, mode="valid"
)
if not os.path.exists("../out"):
    os.makedirs("../out")
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
