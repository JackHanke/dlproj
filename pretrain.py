import torch
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from utils.networks import DemoNet
from utils.losses import combined_loss

class MovesDataSet(Dataset):
    def __init__(self, move_list):
        self.move_list = move_list
    def __len__(self):
        return len(self.move_list)
    def __getitem__(self, idx):
        return self.move_list[idx]

def get_dataloaders(batch_size):
    # read in data, tokenize and make dataloaders
    with open('tests/moves.train.pkl', 'rb') as f:
        train_list = pickle.load(f)
        train_dataset = MovesDataSet(move_list=train_list)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle=True)

    with open('tests/moves.valid.pkl', 'rb') as f:
        valid_list = pickle.load(f)
        valid_dataset = MovesDataSet(move_list=valid_list)
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size = batch_size, shuffle=False)

    return train_dataloader, valid_dataloader


if __name__ == '__main__':
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pretrained_net = DemoNet(num_res_blocks=13)
    pretrained_net.to(device)
    # get dataloaders for each split
    train_dataloader, valid_dataloader = get_dataloaders(batch_size = 128)
    # optimizer and loss
    optim = torch.optim.Adam(pretrained_net.parameters(), lr=0.001, weight_decay=1e-1)

    best_loss = float('inf')
    train_losses, valid_losses = [], []
    train_perps, valid_perps = [], []
    
    epoch = 1
    epochs_with_no_improvement = 0
    while epochs_with_no_improvement <= 8:
        pretrained_net.train()
        
        # train
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch_index, (state, policy, reward) in progress_bar:
            state_batch = state.to(device)
            policy_batch = policy.to(device)
            reward_batch = reward.to(device)
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
            # backprop
            train_loss.backward()
            # update weights
            optim.step()

        train_losses.append(average_loss)
        print(f'Average training loss at Epoch {epoch}: {average_loss}')

        # validation
        with torch.no_grad():
            pretrained_net.eval()
            running_loss = 0.0

            for batch_index, (state, policy, reward, _, _) in enumerate(valid_dataloader):
                state_batch = state.to(device)
                policy_batch = policy.to(device)
                reward_batch = reward.to(device)
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
            # add loss for logging
            valid_losses.append(average_loss)
            print(f'Average validation loss at Epoch {epoch}: {average_loss}')

        # save weights if validation performance is better        
        if average_loss < best_loss:
            best_loss = average_loss
            torch.save(pretrained_net.state_dict(), f'tests/pretrained_model.pth')
            print(' > Model saved!')
        else:
            epochs_with_no_improvement += 1

        epoch += 1

    plt.plot([i+1 for i in range(len(train_losses))], train_losses, label='Training Loss')
    plt.plot([i+1 for i in range(len(valid_losses))], valid_losses, label='Validation Loss')
    plt.xlabel(f'Epochs')
    plt.ylabel(f'Combined Loss')
    plt.title(f'Training Curve for Pretrained dem0')
    plt.legend()
    plt.savefig(f'tests/learning_curve.png')
    plt.show()
