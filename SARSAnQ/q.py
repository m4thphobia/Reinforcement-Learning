if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import gymnasium as gym
from qagent import *
from common.utils import *


if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    num_discretize = 6
    agent = QAgent(env.observation_space.shape[0], env.action_space.n, num_discretize)

    episodes = 300
    penalty = 10
    max_steps = env.spec.max_episode_steps
    episode_rewards = []

    max_steps = env.spec.max_episode_steps
    for episode in range(episodes):
        observation, _ = env.reset()
        state = discretize_state(observation, num_discretize)  # 観測の離散化（状態のインデックスを取得）
        episode_reward = 0
        for t in range(max_steps):
            action = agent.get_action(state, episode)  #  行動を選択
            observation, reward, done, terminated, _ = env.step(action)
            if done and t < max_steps - 1:
                reward = -penalty
            episode_reward += reward
            next_state = discretize_state(observation, num_discretize)
            agent.update_qtable(state, action, reward, next_state)
            state = next_state
            if done or terminated:
                break
        episode_rewards.append(episode_reward)
        if episode % 50 == 0:
            print("Episode %d finished | Episode reward %f" % (episode, episode_reward))

    if not os.path.exists("out"):
        os.makedirs("out")

    plot_moving_average(episode_rewards, num_average_epidodes = 10)

    frames = []
    episodes = 10
    for episode in range(episodes):
        state, info = env.reset()
        state = discretize_state(observation, num_discretize)
        frames.append(env.render())
        for t in range(max_steps):
            action = agent.get_greedy_action(state)
            next_observation, reward, done, terminated, info = env.step(action)
            rendered_frame = env.render()
            if rendered_frame is not None:
                frames.append(rendered_frame)
            state = discretize_state(next_observation, num_discretize)
            if done or terminated:
                break

    env.close()
    save_frames_as_gif(frames)
