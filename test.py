from pettingzoo.classic import chess_v6
from utils.networks import DemoNet
import torch
import time

# NOTE this file tests interaction with the PettingZoo Chess environment

env = chess_v6.env(render_mode="human") # for debugging and human interaction NOTE
# env = chess_v6.env(render_mode=None) # don't render the env NOTE
env.reset(seed=42)

net = DemoNet(num_res_blocks=1)

termination, truncation = False, False
while not termination and not truncation:
    # get state information
    start = time.time()
    observation, reward, termination, truncation, info = env.last()
    print(f'Time to get environment information: {time.time()-start} s')

    if termination or truncation:
        action = None
    else:
        # format obervation for network
        start = time.time()
        state_tensor = torch.tensor(observation['observation'].copy()).float().permute(2, 0, 1)
        state_tensor = state_tensor.unsqueeze(0)
        print(f'Time to format state tensor for network: {time.time()-start} s')

        # compute policy and value
        start = time.time()
        policy, value = net.forward(state_tensor)
        print(f'Time to compute policy: {time.time()-start} s')
        # print(f"Policy Shape: {policy.shape}, Value Shape: {value.shape}")

        # NOTE filter policy vector to legal moves
        start = time.time()
        policy = torch.squeeze(policy)
        # get legal move indices
        legal_moves = torch.tensor(observation['action_mask'])
        legal_indices = torch.nonzero(legal_moves)
        # get maximum value 
        max_val = torch.max(policy[legal_indices])
        # get action that has that maximum value among legal moves
        action = (policy == max_val).nonzero(as_tuple=True)[0].item()
        print(f'Time to get action: {time.time()-start} s')

    # take action
    start = time.time()
    env.step(action)
    print(f'Time to take action: {time.time()-start} s\n')
    
    if action is None: 
        print(f'Reward = {reward}')
        time.sleep(10) # hang on final position for a bit

env.close()
