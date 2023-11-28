import os
import numpy as np
import gymnasium as gym
from agent import QAgent
from utils import(
    discretize_state,
    test_policy,
    save_frames_as_gif,
    plot_moving_average,
)
np.random.seed(1)

# Hyperparameters etc.
NUM_EPISODES = 3000
PENALTY = 10
NUM_DISCRETIZE = 6


def one_episode(env, agent, episode, max_steps):
    observation, _ = env.reset()
    #agent.reset()
    state = discretize_state(observation, NUM_DISCRETIZE)  # 観測の離散化（状態のインデックスを取得）
    episode_reward = 0
    step = 0
    for t in range(max_steps):
        action = agent.get_action(state, episode)
        observation, reward, done, truncated, _ = env.step(action)
        next_state = discretize_state(observation, NUM_DISCRETIZE)
        if done and t < max_steps - 1:
            reward = -PENALTY

        episode_reward += reward
        agent.update_qtable(state, action, reward, done, next_state)

        state = next_state
        step += 1
        if done or truncated:
            break

    return episode_reward, step


def main():

    if not os.path.exists("../out"):
        os.makedirs("../out")

    env = gym.make('CartPole-v1', render_mode="rgb_array")
    agent = QAgent(env.observation_space.shape[0], env.action_space.n, NUM_DISCRETIZE)

    max_steps = env.spec.max_episode_steps
    episode_rewards = []

    for episode in range(NUM_EPISODES):

        episode_reward, step = one_episode(env, agent, episode, max_steps)
        episode_rewards.append(episode_reward)

        if episode % 100 == 0:
            print(f"Episode {int(episode/NUM_EPISODES*100)}% done | Episode reward: {episode_reward} | Step: {step}")


    plot_moving_average(episode_rewards, num_average_epidodes = 10)
    frames = test_policy(env, agent, NUM_DISCRETIZE, max_steps)
    save_frames_as_gif(frames)


if __name__ == '__main__':
    main()
