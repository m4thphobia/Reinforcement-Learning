import os
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import animation


def save_frames_as_gif(frames, path="../out", filename=f'/{os.path.basename(__file__).split(".")[0]}.gif'):

    if not os.path.exists(path):
        os.makedirs(path)

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


env = gym.make('CartPole-v1', render_mode="rgb_array")
frames = []
episodes = 3
for episode in range(episodes):
    state, info = env.reset()
    done = False
    truncated = False
    while not done and not truncated:
        rendered_frame = env.render()
        if rendered_frame is not None:
            frames.append(rendered_frame)
            print(f"appended: {rendered_frame.shape}")
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)



env.close()
save_frames_as_gif(frames)



