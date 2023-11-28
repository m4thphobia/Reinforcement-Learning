if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import gymnasium as gym
from common.utils import *

env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")

frames = []
episodes = 1
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    truncated = False
    while not done and not truncated:
        rendered_frame = env.render()
        if rendered_frame is not None:
            frames.append(rendered_frame)
        action = env.action_space.sample()
        print(action)
        next_state, reward, done, truncated, info = env.step(action)

env.close()
save_frames_as_gif(frames)
