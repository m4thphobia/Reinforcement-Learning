import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation


def set_seed(seed: int=42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

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

def save_frames_as_gif(frames, path="../out/", filename="actor_critic_LunarLander-v2.gif"):

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

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Moving Average")
    ax.plot(np.arange(len(moving_average)), moving_average)
    ax.set_xlabel("episode")
    ax.set_ylabel("rewards")
    plt.savefig(path + filename)


def save_checkpoint(state, filename="../var/my_filename.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
