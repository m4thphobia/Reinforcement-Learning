import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import gymnasium as gym
from agent import *
from utils import(
    test_policy,
    save_frames_as_gif,
    plot_moving_average,
    save_checkpoint,
    load_checkpoint,
)
np.random.seed(1)

# Hyperparameters etc.
SAVE_PATH = "../var/my_filename.pth.tar"
LOAD_MODEL = False

NUM_EPISODES = 10000
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
MEMORY_SIZE = 50000
INITIAL_MEMORY_SIZE = 500

def one_episode(env, agent, episode, max_steps):
    state, _ = env.reset()
    episode_reward = 0
    for t in range(max_steps):
        action = agent.get_action(state, episode)
        next_state, reward, done, truncated, _ = env.step(action)

        episode_reward += reward

        transition = {
            "state": state,
            "reward": reward,
            "action": action,
            "next_state": next_state,
            "done": int(done),
        }
        agent.replay_buffer.append(transition)
        agent.update_q()

        state = next_state
        if done or truncated:
            break

    return episode_reward

def memory_initializer(env, agent):
    state, _ = env.reset()
    for step in range(INITIAL_MEMORY_SIZE):
        action = env.action_space.sample()
        next_state, reward, done, truncated, _ = env.step(action)
        transition = {
            "state": state,
            "reward": reward,
            "action": action,
            "next_state": next_state,
            "done": int(done),
        }
        agent.replay_buffer.append(transition)
        state, _ = env.reset() if done or truncated else (next_state, _)


def main():

    if not os.path.exists("../out"):
        os.makedirs("../out")

    if not os.path.exists("../var"):
        os.makedirs("../var")

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    max_steps = env.spec.max_episode_steps
    agent = DqnAgent(env.observation_space.shape[0], env.action_space.n, memory_size=MEMORY_SIZE)

    if LOAD_MODEL:
        load_checkpoint(torch.load(SAVE_PATH), agent.qnet, agent.optimizer)

    memory_initializer(env, agent)

    episode_rewards = []
    for episode in range(NUM_EPISODES):
        episode_reward = one_episode(env, agent, episode, max_steps)
        episode_rewards.append(episode_reward)

        if episode % 1000 == 0:
            checkpoint = {
                "state_dict": agent.qnet.state_dict(),
                "optimizer": agent.optimizer.state_dict(),
            }

            save_checkpoint(checkpoint, SAVE_PATH)

        if episode % 100 == 0:
            print(f"Episode {int(episode/NUM_EPISODES*100)}% done | Episode reward: {episode_reward}")

    save_checkpoint(checkpoint, SAVE_PATH)
    plot_moving_average(episode_rewards, 50)
    frames = test_policy(env, agent, max_steps)
    save_frames_as_gif(frames)

if __name__ == '__main__':
    main()
