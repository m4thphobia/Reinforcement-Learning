import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def bins(clip_min: float, clip_max: float, num: float) -> np.ndarray:
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]


def discretize_state(observation: np.ndarray, num_discretize: int) -> int:
    c_pos, c_v, p_angle, p_v = observation
    discretized = [
        np.digitize(c_pos, bins=bins(-2.4, 2.4, num_discretize)),
        np.digitize(c_v, bins=bins(-3.0, 3.0, num_discretize)),
        np.digitize(p_angle, bins=bins(-0.5, 0.5, num_discretize)),
        np.digitize(p_v, bins=bins(-2.0, 2.0, num_discretize)),
    ]
    return sum([x * (num_discretize**i) for i, x in enumerate(discretized)])


def test_policy(env, agent, num_discretize, max_steps):

    frames = []
    episodes = 1
    for episode in range(episodes):
        observation, _ = env.reset()
        state = discretize_state(observation, num_discretize)
        frames.append(env.render())
        for t in range(max_steps):
            action = agent.get_greedy_action(state)
            next_observation, reward, done, truncated, _ = env.step(action)
            rendered_frame = env.render()
            assert rendered_frame is not None, "render can't be None"
            frames.append(rendered_frame)
            state = discretize_state(next_observation, num_discretize)
            if done or truncated:
                break

    env.close()
    return frames

def save_frames_as_gif(frames, path="../out/", filename="sarsa_cartpole.gif"):

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
