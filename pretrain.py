import torch
import pickle
import chess
import time

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

from utils.networks import DemoNet
from utils.losses import combined_loss
from utils.chess_utils_local import get_observation
from utils.utils import prepare_state_for_net, observe
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
from utils.training import Checkpoint


def train_val_split():
    training_data = []
    folder_path = Path('data')
    pickle_files = sorted(folder_path.glob("*.pkl"), key=lambda f: f.name)

    for filepath in pickle_files:
        i = str(filepath).split('-')[-1].removesuffix(".pkl")
        print(f'Reading in data {i}')
        with filepath.open('rb') as f:
            training_data += pickle.load(f)
        
    train, valid = train_test_split(training_data, test_size=0.2, random_state=42)
    print(f'Length of train: {len(train)}, Length of validation: {len(valid)}')
    return train, valid


class MovesDataSet(Dataset):
    def __init__(self, move_list):
        self.move_list = move_list

    def __len__(self):
        return len(self.move_list)
    
    def __getitem__(self, idx):
        board, policy, z, player_string, bh = self.move_list[idx]
        obs = observe(
            board=board,
            agent=player_string,
            possible_agents=['player_0', 'player_1'],
            board_history=bh
        )
        state = obs['observation'].copy()
        z = torch.tensor([z])
        policy_vec = torch.zeros(4672).float()
        policy_vec[policy] = 1.0
        return prepare_state_for_net(state).squeeze(), (policy_vec, z)


def get_dataloaders(train_list, valid_list, batch_size):
    # read in data, tokenize and make dataloaders
    train_dataset = MovesDataSet(move_list=train_list)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle=True)

    valid_dataset = MovesDataSet(move_list=valid_list)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size = batch_size, shuffle=False)
    return train_dataloader, valid_dataloader



def pretrain():
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    pretrained_net = DemoNet(num_res_blocks=13)
    pretrained_net.to(device)
    # get dataloaders for each split
    train_list, val_list = train_val_split()
    train_ds = MovesDataSet(train_list)
    train_dataloader, valid_dataloader = get_dataloaders(train_list=train_list, valid_list=val_list, batch_size = 128)
    # optimizer and loss
    optim = torch.optim.Adam(
        pretrained_net.parameters(), 
        lr=0.001, 
        weight_decay=1e-4
    )

    best_loss = float('inf')
    train_losses, valid_losses = [], []
    train_perps, valid_perps = [], []
    
    epoch = 1
    epochs_with_no_improvement = 0
    while epochs_with_no_improvement <= 3:
        print(f"Epoch {epoch}:")
        pretrained_net.train()
        
        # train
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch_index, (state_batch, (policy_batch, reward_batch)) in progress_bar:
            # send to device
            state_batch = state_batch.to(device)
            policy_batch = policy_batch.to(device)
            reward_batch = reward_batch.to(device)
            # zero gradients
            optim.zero_grad()
            # get prediction and hidden state from RNN 
            policy_out, value_out = pretrained_net(state_batch)

            # loss calc
            train_loss = combined_loss(
                pi=policy_batch, 
                p_theta_logits=policy_out, 
                z=reward_batch, 
                v_theta=value_out, 
                policy_weight=1.0, 
                value_weight=1.0
            )


            running_loss += train_loss.item()
            average_loss = running_loss / (batch_index + 1)
            progress_bar.set_description(f"Average Train Loss: {average_loss:.4f}")
            # backprop
            train_loss.backward()
            # update weights
            optim.step()

        train_losses.append(average_loss)

        # validation
        with torch.no_grad():
            pretrained_net.eval()
            running_loss = 0.0

            progress_bar = tqdm(enumerate(valid_dataloader), total=len(valid_dataloader))
            for batch_index, (state_batch, (policy_batch, reward_batch)) in progress_bar:
                # send to device
                state_batch = state_batch.to(device)
                policy_batch = policy_batch.to(device)
                reward_batch = reward_batch.to(device)
                # val prediction
                policy_out, value_out = pretrained_net(state_batch)
                # loss calc
                valid_loss = combined_loss(
                    pi=policy_batch, 
                    p_theta_logits=policy_out, 
                    z=reward_batch, 
                    v_theta=value_out, 
                    policy_weight=1.0, 
                    value_weight=1.0
                )
                running_loss += valid_loss.item()
                average_loss = running_loss / (batch_index + 1)
                progress_bar.set_description(f"Average Val Loss: {average_loss:.4f}")
            # add loss for logging
            valid_losses.append(average_loss)

        # save weights if validation performance is better        
        if average_loss < best_loss:
            best_loss = average_loss
            torch.save(pretrained_net.state_dict(), f'tests/pretrained_model.pth')
            print(' > Model saved!')
            epochs_with_no_improvement = 0
        else:
            epochs_with_no_improvement += 1

        epoch += 1
        print('\n')

    plt.plot([i+1 for i in range(len(train_losses))], train_losses, label='Training Loss')
    plt.plot([i+1 for i in range(len(valid_losses))], valid_losses, label='Validation Loss')
    plt.xlabel(f'Epochs')
    plt.ylabel(f'Combined Loss')
    plt.title(f'Training Curve for Pretrained dem0')
    plt.legend()
    plt.savefig(f'tests/learning_curve.png')
    plt.show()


def evaluate():
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    client = Checkpoint(verbose=True, compute_elo=False)
    state_dict = client.download_from_blob("checkpoints/pretrained_weights.pth", device=device)

    pretrained_net = DemoNet(num_res_blocks=13)
    pretrained_net.to(device)
    pretrained_net.load_state_dict(state_dict=state_dict)
    # get dataloaders for each split
    train_list, val_list = train_val_split()
    _, valid_dataloader = get_dataloaders(train_list=train_list, valid_list=val_list, batch_size = 128)
    with torch.no_grad():
        pretrained_net.eval()
        running_loss = 0.0

        progress_bar = tqdm(enumerate(valid_dataloader), total=len(valid_dataloader))
        for batch_index, (state_batch, (policy_batch, reward_batch)) in progress_bar:
            # send to device
            state_batch = state_batch.to(device)
            policy_batch = policy_batch.to(device)
            reward_batch = reward_batch.to(device)
            # val prediction
            policy_out, value_out = pretrained_net(state_batch)
            # loss calc
            valid_loss = combined_loss(
                pi=policy_batch, 
                p_theta_logits=policy_out, 
                z=reward_batch, 
                v_theta=value_out, 
                policy_weight=1.0, 
                value_weight=1.0
            )
            running_loss += valid_loss.item()
            average_loss = running_loss / (batch_index + 1)
            progress_bar.set_description(f"Average Val Loss: {average_loss:.4f}")
    

if __name__ == '__main__':
    evaluate()