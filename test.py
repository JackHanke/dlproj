from pettingzoo.classic import chess_v6
from utils.networks import DemoNet
import torch

env = chess_v6.env(render_mode="human")
env.reset(seed=42)
net = DemoNet(num_res_blocks=1)

observation, reward, termination, truncation, info = env.last()
policy, value = net.forward(torch.tensor(observation['observation']).float().permute(2, 0, 1).unsqueeze(0))
print(f"Policy Shape: {policy.shape}, Value Shape: {value.shape}")