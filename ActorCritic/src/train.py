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

NUM_EPISODES = 3000
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def one_episode(env, agent, max_steps):
    state, _ = env.reset()
    episode_reward = 0
    n_step = 0
    for t in range(max_steps):
        action, prob, state_value = agent.get_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        agent.add_memory(reward, prob, state_value)

        state = next_state
        episode_reward += reward
        n_step += 1

        if done or truncated:
            agent.update_policy()
            break

    return episode_reward, n_step


def main():

    if not os.path.exists("../out"):
        os.makedirs("../out")

    if not os.path.exists("../var"):
        os.makedirs("../var")

    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    max_steps = env.spec.max_episode_steps
    if max_steps == None:
        max_steps = 100000000000

    agent = ActorCriticAgent(env.observation_space.shape[0], env.action_space.n)

    if LOAD_MODEL:
        load_checkpoint(torch.load(SAVE_PATH), agent.acnet, agent.optimizer)

    episode_rewards = []
    for episode in range(NUM_EPISODES):
        episode_reward, n_step = one_episode(env, agent, max_steps)
        episode_rewards.append(episode_reward)

        if episode % 1000 == 0 or episode == (NUM_EPISODES - 1):
            checkpoint = {
                "state_dict": agent.acnet.state_dict(),
                "optimizer": agent.optimizer.state_dict(),
            }

            save_checkpoint(checkpoint, SAVE_PATH)

        if episode % 100 == 0:
            print(f"Episode {int(episode/NUM_EPISODES*100)}% done | Episode reward: {episode_reward} | Steps :{n_step}")

    plot_moving_average(episode_rewards, 50)
    frames = test_policy(env, agent, max_steps)
    save_frames_as_gif(frames)

if __name__ == '__main__':
    main()
