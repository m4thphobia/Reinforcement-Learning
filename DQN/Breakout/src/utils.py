import numpy as np
import random
import torch
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt
from matplotlib import animation
import cv2

def set_seed(seed: int=42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, filename="../var/my_filename.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(self, env=None, repeat=4, clip_reward=False, no_ops=0,
                fire_first=False):
        super().__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.frame_buffer = deque([], maxlen=2)
        self.clip_reward = clip_reward
        self.no_ops = no_ops
        self.fire_first = fire_first

    def step(self, action):
        t_reward = 0.0
        done = False
        for i in range(self.repeat):
            obs, reward, done, truncated, info = self.env.step(action)
            if self.clip_reward:
                reward = np.clip(np.array([reward]), -1, 1)[0]
            t_reward += reward
            self.frame_buffer.append(obs)
            if done:
                break

        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, t_reward, done, truncated, info

    # def reset(self):
    #     obs, info = self.env.reset()
    #     no_ops = np.random.randint(self.no_ops)+1 if self.no_ops > 0 else 0
    #     for _ in range(no_ops):
    #         _, _, done, truncated, _ = self.env.step(0)
    #         if done or truncated:
    #             self.env.reset()
    #     if self.fire_first:
    #         assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
    #         obs, _, _, _, _ = self.env.step(1)
    #     self.frame_buffer = deque([], maxlen=2)
    #     self.frame_buffer[0] = obs

    #     return obs, info


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env=None):
        super().__init__(env)
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                    shape=self.shape, dtype=np.float32)

    def observation(self, obs):
        new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(new_frame, self.shape[1:],
                                    interpolation=cv2.INTER_AREA)
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        new_obs = new_obs / 255.0

        return new_obs


class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
                            env.observation_space.low.repeat(repeat, axis=0),
                            env.observation_space.high.repeat(repeat, axis=0),
                            dtype=np.float32)
        self.stack = deque(maxlen=repeat)

    def reset(self):
        self.stack.clear()
        observation, info = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape), info

    def observation(self, observation):
        self.stack.append(observation)
        return np.array(self.stack).reshape(self.observation_space.low.shape)


def make_env(env_name, shape=(84,84,1), repeat=4, clip_rewards=False,
            no_ops=0, fire_first=False):

    env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env, repeat, clip_rewards, no_ops, fire_first)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, repeat)

    return env


def test_policy(env, agent, max_steps):

    frames = []
    episodes = 1
    for episode in range(episodes):
        state, _ = env.reset()
        frames.append(env.render())
        for t in range(max_steps):
            action = agent.get_greedy_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            rendered_frame = env.render()
            assert rendered_frame is not None, "render can't be None"
            frames.append(rendered_frame)
            state = next_state
            if done or truncated:
                break

    env.close()
    return frames


def save_frames_as_gif(frames, path="../out/", filename="dqn_breakout.gif"):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


def plot_moving_average(episode_rewards, num_average_epidodes=10 , path="../out/", filename="average_reward.png"):

    moving_average = np.convolve(episode_rewards, np.ones(num_average_epidodes) / num_average_epidodes, mode="valid")

    fig = plt.figure() # Figureを作成
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Moving Average")
    ax.plot(np.arange(len(moving_average)), moving_average)
    ax.set_xlabel("episode")
    ax.set_ylabel("rewards")
    plt.savefig(path + filename)


def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time
