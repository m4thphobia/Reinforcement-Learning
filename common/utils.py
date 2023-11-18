import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import torch


def set_seed(seed: int) -> None:
#    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_frames_as_gif(frames, path="out", filename=f'/{sys.argv[0].split(".")[0]}.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


def plot_moving_average(episode_rewards, num_average_epidodes=10 , path="out", filename=f'/{sys.argv[0].split(".")[0]}.png'):

    moving_average = np.convolve(episode_rewards, np.ones(num_average_epidodes) / num_average_epidodes, mode="valid")

    fig = plt.figure() # Figureを作成
    ax = fig.add_subplot(1,1,1)
    ax.set_title(f"{os.path.basename(__file__).split('.')[0]}")
    ax.plot(np.arange(len(moving_average)), moving_average)
    ax.set_xlabel("episode")
    ax.set_ylabel("rewards")
    plt.savefig(path + filename)


if __name__ == '__main__':
    if not os.path.exists("../out"):
        os.makedirs("../out")
