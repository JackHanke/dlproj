from utils.networks import DemoNet
from utils.self_play import SelfPlayAgent
from utils.losses import combined_loss
from utils.memory import ReplayMemory
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Union


def train_on_replay_memory(
    data: ReplayMemory, 
    network: DemoNet, 
    batch_size: int, 
    device: torch.device, 
    optimizer: Union[optim.Adam, optim.SGD]
):
    if len(data) < batch_size:
        return
    # Zero grad
    optimizer.zero_grad()
    # Get batch
    batch_dict = data.sample_in_batches(batch_size=batch_size)
    state_batch = batch_dict['state_batch'].to(device)
    policy_batch = batch_dict['policy_batch'].to(device)
    reward_batch = batch_dict['reward_batch'].to(device)

    # Forward pass
    policy_out, value_out = network(state_batch)
    loss = combined_loss(pi=policy_batch, p_theta_logits=policy_out, z=reward_batch, v_theta=value_out)

    # Backward pass
    loss.backward()
    optimizer.step()