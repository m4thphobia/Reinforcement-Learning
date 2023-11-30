import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import gymnasium as gym
from tqdm import tqdm
from timeit import default_timer as timer
from agent import *
from utils import(
    test_policy,
    save_frames_as_gif,
    plot_moving_average,
    save_checkpoint,
    load_checkpoint,
    set_seed,
    make_env,
    print_train_time
)

set_seed()

# Hyperparameters etc.
SAVE_PATH = "../var/my_filename.pth.tar"
LOAD_MODEL = False

NUM_EPISODES = 500
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
MEMORY_SIZE = 50000
INITIAL_MEMORY_SIZE = 500


def one_episode(env, agent, max_steps):
    state, _ = env.reset()
    episode_reward = 0
    n_step = 0
    for t in range(max_steps):
        action = agent.get_action(state)
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
        n_step += 1

        if done or truncated:
            break

    return episode_reward, n_step

def memory_initializer(env, agent, initial_memory_size=INITIAL_MEMORY_SIZE):
    state, _ = env.reset()
    for step in range(initial_memory_size):
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

    env = make_env("PongNoFrameskip-v4")
    best_score = -np.inf
    max_steps = env.spec.max_episode_steps
    if max_steps == None:
        max_steps = 100000000000

    agent = DqnAgent(env.observation_space.shape, env.action_space.n)

    if LOAD_MODEL:
        load_checkpoint(torch.load(SAVE_PATH), agent.qnet, agent.optimizer)

    memory_initializer(env, agent)

    episode_rewards = []
    steps = []
    start = timer()
    for episode in tqdm(range(NUM_EPISODES)):
        episode_reward, n_step = one_episode(env, agent, max_steps)
        episode_rewards.append(episode_reward)
        steps.append(n_step)

        if episode % 1000 == 0:
            checkpoint = {
                "state_dict": agent.qnet.state_dict(),
                "optimizer": agent.optimizer.state_dict(),
            }

            save_checkpoint(checkpoint, SAVE_PATH)

        if episode % 100 == 0:
            print(f"Episode {int(episode/NUM_EPISODES*100)}% done | Episode reward: {episode_reward} | Steps :{n_step}")
    end = timer()
    print_train_time(start, end)
    save_checkpoint(checkpoint, SAVE_PATH)
    plot_moving_average(episode_rewards, 50)
    frames = test_policy(env, agent, max_steps)
    save_frames_as_gif(frames)

if __name__ == '__main__':
    main()
