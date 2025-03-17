from pettingzoo.classic import chess_v6, tictactoe_v3
import torch
import torch.nn as nn
import time
import chess
import numpy as np
from copy import deepcopy
import torch.multiprocessing as mp


from utils.utils import prepare_state_for_net, get_net_best_legal
from utils.chess_utils_local import action_to_move
from utils.self_play import SelfPlaySession
from utils.memory import ReplayMemory, Transition
from utils.utils import Timer

from ttt_utils.ttt_evaluator import evaluator
from ttt_utils.ttt_agent import RandomAgent, Human, Agent, PerfectAgent
from ttt_utils.ttt_networks import DemoTicTacToeConvNet, DemoTicTacToeFeedForwardNet

# NOTE this file tests interaction with the PettingZoo Chess environment


def test_self_play():
    network = DemoNet(num_res_blocks=1)
    session = SelfPlaySession()
    replay_memory = ReplayMemory(1000)
    session.run_self_play(
        training_data=replay_memory,
        network=network,
        n_sims=3,
        num_games=4,
        max_moves=100
    )


def test_env():
    env = chess_v6.env()
    env.reset()
    observation, reward, termination, truncation, info = env.last()
    print("No moves taken yet...")
    print(termination, truncation)
    current_player = env.agent_selection
    print(f"Current Player {current_player}")
    env.step(77)
    print(env.terminations[current_player], env.truncations[current_player])


def test_replay_memory():
    env = chess_v6.env()
    env.reset()
    observation, reward, termination, truncation, info = env.last()
    state = observation['observation']
    memory = ReplayMemory(1000)
    for _ in range(32):
        memory.push(torch.from_numpy(state), torch.randn(4672,), torch.tensor([1]))

    print(len(memory))
    sampled_memory = memory.sample(10)
    batch = Transition(*zip(*sampled_memory))
    state_batch = torch.stack(batch.state)
    policy_batch = torch.stack(batch.policy)
    result_batch = torch.stack(batch.game_result)
    print(state_batch.shape, policy_batch.shape, result_batch.shape)
    print(len(memory))


def test_evaluator():
    agent_1 = RandomAgent()
    agent_2 = PerfectAgent()

    device = torch.device('cpu')

    wins, draws, losses, win_percent, tot_games = evaluator(
        challenger_agent=agent_1,
        current_best_agent=agent_2,
        device=device,
        max_moves=10,
        num_games=100,
        v_resign=-1.5,
        external = True
    )
    print(f'Against {agent_2.name}, won {wins} games, drew {draws} games, lost {losses} games. ({round(100*win_percent, 2)}% wins, {round(100*(draws/tot_games), 2)}% draws)')




def test(verbose=False):
    # env = chess_v6.env(render_mode="human") # for debugging and human interaction NOTE
    env = tictactoe_v3.env(render_mode="human") # for debugging and human interaction NOTE
    # env = chess_v6.env(render_mode=None) # don't render the env NOTE
    env.reset()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    net = DemoTicTacToeConvNet(
        input_channels=2,
        num_res_blocks=1,
        policy_output_dim=9
    ).to(device)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Number of parameters: {total_params}")

    agent_1 = Agent(
        version=0,
        network=net,
        sims=15
    )
    # agent_1 = PerfectAgent()
    # agent_1 = RandomAgent()
    # agent_1 = Human()

    # agent_2 = Agent(
    #     version=0,
    #     network=net,
    #     sims=10
    # )
    # agent_2 = PerfectAgent()
    # agent_2 = RandomAgent()
    agent_2 = Human()

    i = 0
    times = []
    termination, truncation = False, False
    while not termination and not truncation:
        env.render()

        # print(env.board)
        # print(type(env.board))
        # input()

        # get state information
        start = time.time()
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            # format obervation for network
            start = time.time()
            state_tensor = prepare_state_for_net(state=observation['observation'].copy()).to(device)
            # if verbose: print(f'Time to format state tensor for network: {time.time()-start} s')

            # compute policy and value
            start = time.time()
            # policy, value = net.forward(state_tensor)
            if verbose: 
                # print(f"Policy Shape: {policy.shape}, Value Shape: {value.shape}")
                print(f'Time to compute policy: {time.time()-start} s')

            start = time.time()
            sims = 100
            num_threads = 4

            if (i % 2) == 0:
                action, val = agent_1.inference(
                    observation=observation,
                    board=deepcopy(env.board),
                    device=torch.device('cpu')
                )

            elif (i % 2) == 1:
                action, val = agent_2.inference(
                    observation=observation,
                    board=deepcopy(env.board),
                    device=torch.device('cpu')
                )


            t = time.time()-start
            print(f'MCTS with {sims} sims completes after {t} s')
            times.append(t)
            i += 1
            # input()

        # take action
        start = time.time()
        env.step(action)
        print(f'env board squares: {env.board.squares}')
        if verbose: print(f'Time to take action: {time.time()-start} s\n')
        
        if action is None: 
            if verbose: print(f'Reward = {reward}')
            time.sleep(10) # hang on final position for a bit

    env.close()
    print(f'Average MCTS (sims: {sims} threads: {num_threads}) time: {sum(times)/len(times)} s')

if __name__ == '__main__':

    test(verbose=True)

    # test_evaluator()

    