from pettingzoo.classic import chess_v6
from utils.networks import DemoNet
import torch
import time
from utils.mcts import mcts
from utils.vanillamcts import vanilla_mcts
from utils.utils import prepare_state_for_net, get_net_best_legal
from utils.chess_utils_local import action_to_move
from copy import deepcopy
import numpy as np
from sys import platform
import os

# NOTE this file tests interaction with the PettingZoo Chess environment

def test(verbose=False):
    # env = chess_v6.env(render_mode="human") # for debugging and human interaction NOTE
    env = chess_v6.env(render_mode=None) # don't render the env NOTE
    env.reset(seed=42)

    net = DemoNet(num_res_blocks=1)

    termination, truncation = False, False
    while not termination and not truncation:
        if verbose: print(env.board)
        # get state information
        start = time.time()
        observation, reward, termination, truncation, info = env.last()
        if verbose: print(f'Time to get environment information: {time.time()-start} s')

        if termination or truncation:
            action = None
        else:
            # format obervation for network
            start = time.time()
            state_tensor = prepare_state_for_net(state=observation['observation'].copy())
            if verbose: print(f'Time to format state tensor for network: {time.time()-start} s')

            # compute policy and value
            start = time.time()
            policy, value = net.forward(state_tensor)
            if verbose: 
                # print(f"Policy Shape: {policy.shape}, Value Shape: {value.shape}")
                print(f'Time to compute policy: {time.time()-start} s')

            start = time.time()
            sims = 10
            pi, val, action = mcts(state=deepcopy(env.board), net=net, tau=1, sims=sims, verbose=False)
            if verbose: print(f'MCTS with {sims} sims completes after {time.time()-start} s')
            input()

        # take action
        start = time.time()
        env.step(action)
        if verbose: print(f'Time to take action: {time.time()-start} s\n')
        
        if action is None: 
            if verbose: print(f'Reward = {reward}')
            time.sleep(10) # hang on final position for a bit

    env.close()

if __name__ == '__main__':
    test(verbose=True)

