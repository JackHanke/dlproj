from pettingzoo.classic import chess_v6
from utils.networks import DemoNet
import torch
import time
from utils.mcts import mcts
from utils.utils import prepare_state_for_net, get_net_best_legal
from copy import deepcopy

# NOTE this file tests interaction with the PettingZoo Chess environment

# env = chess_v6.env(render_mode="human") # for debugging and human interaction NOTE
env = chess_v6.env(render_mode=None) # don't render the env NOTE
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
        state_tensor = prepare_state_for_net(state=observation['observation'].copy())
        print(f'Time to format state tensor for network: {time.time()-start} s')

        print(env.board)

        # compute policy and value
        start = time.time()
        policy, value = net.forward(state_tensor)
        print(f'Time to compute policy: {time.time()-start} s')
        print(f"Policy Shape: {policy.shape}, Value Shape: {value.shape}")

        sims = 2
        start = time.time()
        # NOTE state is the python-chess board obj env.board, not the observation obj
        mcts(state=deepcopy(env.board), net=net, tau=1, sims=sims)
        print(f'MCTS with {sims} sims completes after {time.time()-start} s')

        # NOTE filter policy vector to legal moves
        start = time.time()
        action = get_net_best_legal(policy_vec=policy, legal_moves=observation['action_mask'])
        print(f'Time to get action: {time.time()-start} s')

    # take action
    start = time.time()
    env.step(action)
    print(f'Time to take action: {time.time()-start} s\n')
    
    if action is None: 
        print(f'Reward = {reward}')
        time.sleep(10) # hang on final position for a bit

env.close()
