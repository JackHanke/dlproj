from utils.networks import DemoNet
from utils.self_play import SelfPlaySession
from utils.agent import Agent
from utils.losses import combined_loss
from utils.memory import ReplayMemory
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Union
import json
import os
import torch.multiprocessing as mp


def train_on_batch(
    data: ReplayMemory, 
    network: DemoNet, 
    batch_size: int, 
    device: torch.device, 
    optimizer: Union[optim.Adam, optim.SGD]
):
    network.train()
    if len(data) <= batch_size:
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


def _background_compute_elo_and_save(agent: Agent, info_path: str, verbose: str) -> None:
    if verbose:
        print("Beginning Elo computation in background process...")
    agent.compute_bayes_elo()  # This is the slow function
    # Prepare info to save
    info = {
        "version": agent.version,
        "elo": agent.elo
    }
    # Ensure the directory exists
    os.makedirs(os.path.dirname(info_path), exist_ok=True)
    with open(info_path, "w") as f:
        json.dump(info, f)
    if verbose:
        print(f"Finished Elo computation and saved info to {info_path}")


class Checkpoint:
    def __init__(self, verbose: bool, compute_elo: bool = True):
        self.compute_elo = compute_elo
        self.verbose = verbose
        self.best_agent = None
        self.best_weights = None
        self.elo = None
        self.version = -1
        self.iteration = None

    def save_state_dict(self, path: str) -> None:
        assert path.split(".")[-1] == "pth", f"Must be in .pth format. Your current path is {path}."
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.best_weights, path)
        if self.verbose:
            print(f"Saved new best weights to {path}")

    def step(self, weights_path: str, info_path: str, current_best_agent: Agent):
        assert info_path.split(".")[-1] == 'json', "Info path must be in JSON format."
        if current_best_agent.version > self.version:
            self.save_state_dict(path=weights_path)
            self.best_agent = current_best_agent
            self.best_weights = current_best_agent.network.state_dict()
            self.version = current_best_agent.version

            if self.compute_elo:
                if self.verbose:
                    print("Starting background process for Elo computation...")

                # Start the Elo computation in a separate process
                p = mp.Process(
                    target=_background_compute_elo_and_save,
                    args=(current_best_agent, info_path, self.verbose)
                )
                p.start() # Will run in background since we dont have a .join()
            else:
                info = {
                    "version": current_best_agent.version
                }
                # Ensure the directory exists
                os.makedirs(os.path.dirname(info_path), exist_ok=True)
                with open(info_path, "w") as f:
                    json.dump(info, f)
                if self.verbose:
                    print(f"Finished Elo computation and saved info to {info_path}")
