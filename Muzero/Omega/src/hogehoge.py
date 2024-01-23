import config
import torch
from network import *


def sample_encoded_state(n=1, threshold = 0.5):
    sampled_vector = torch.randn((n,128))
    sampled_tensor = torch.tensor(sampled_vector).view(n, config.num_hidden, config.board_length, config.board_length)

    network = Network()
    network.state_dict(torch.load('../var/champion.pth'))
    sampled_state = network.vae.decode(sampled_tensor)[0:2] # third layer means nothing here

    binary_tensor = torch.where(sampled_state > threshold, torch.tensor(1.0), torch.tensor(0.0))

    return sampled_state

def draw_legal_reconstruction_rate():
    pass