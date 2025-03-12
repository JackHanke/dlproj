from pettingzoo.classic import chess_v6
from utils.networks import DemoNet
import torch
import time
# from utils.mcts import mcts
# from utils.mcts_mp_danya import parallel_mcts
from utils.mcts import mcts


# from utils.vanillamcts import vanilla_mcts
from utils.utils import prepare_state_for_net, get_net_best_legal
from utils.chess_utils_local import action_to_move
from copy import deepcopy
import numpy as np
from utils.agent import Agent
from utils.self_play import SelfPlaySession
from utils.memory import ReplayMemory, Transition
from utils.evaluator import evaluator
import chess
import torch.multiprocessing as mp
from utils.utils import Timer


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


def test(verbose=False):
    env = chess_v6.env(render_mode="human") # for debugging and human interaction NOTE
    # env = chess_v6.env(render_mode=None) # don't render the env NOTE
    env.reset(seed=42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    net = DemoNet(num_res_blocks=1).to(device)
    # net.share_memory()
    times = []
    termination, truncation = False, False
    while not termination and not truncation:
        if verbose: print(env.board)
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
            policy, value = net.forward(state_tensor)
            if verbose: 
                # print(f"Policy Shape: {policy.shape}, Value Shape: {value.shape}")
                print(f'Time to compute policy: {time.time()-start} s')

            start = time.time()
            sims = 100
            num_threads = 1
            # with Timer():
            pi, val, action = mcts(
                state=deepcopy(env.board), 
                observation=observation['observation'],
                net=net, 
                tau=1, 
                sims=sims, 
                num_threads=num_threads,
                device=device, 
                verbose=False
            )
            # print("Final policy vector (pi):", pi)
            # print("Estimated value:", val)
            print("Chosen action index:", action)
            t = time.time()-start
            if verbose: print(f'MCTS with {sims} sims completes after {t} s')
            times.append(t)
            # input()

        # take action
        start = time.time()
        env.step(action)
        if verbose: print(f'Time to take action: {time.time()-start} s\n')
        
        if action is None: 
            if verbose: print(f'Reward = {reward}')
            time.sleep(10) # hang on final position for a bit

    env.close()
    print(f'Average MCTS (sims: {sims} threads: {num_threads}) time: {mean(times)} s')

if __name__ == '__main__':
    # test(verbose=True)
    # test_mcts_parallel()
    # agent_1 = Agent(version=1, network=DemoNet(num_res_blocks=1))
    # agent_2 = Agent(version=2, network=DemoNet(num_res_blocks=1))

    # winner_agent = evaluator(challenger_agent=agent_1, current_best_agent=agent_2)
    # print(f'Winner agent is Agent {winner_agent.version}')

    # from utils.evaluator import evaluator

    current_best_version = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_best_network = DemoNet(num_res_blocks=1).to(device)
    challenger_network = deepcopy(current_best_network).to(device)
    current_best_agent = Agent(
        version=current_best_version, 
        network=current_best_network, 
        sims=5
    )
    challenger_agent = Agent(
        version=current_best_version+1, 
        network=challenger_network, 
        sims=5
    )

    current_best_agent = evaluator(
        challenger_agent=challenger_agent, 
        current_best_agent=current_best_agent,
        device=device,
        max_moves=100,
        num_games=3,
        v_resign=-0.95, 
        verbose=True
    )    