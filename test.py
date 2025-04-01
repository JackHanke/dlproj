from pettingzoo.classic import chess_v6
import torch
import time
import chess
import numpy as np
# from utils.mcts import mcts
# from utils.mcts_mp_danya import parallel_mcts
from copy import deepcopy
from utils.mcts import mcts
import torch.multiprocessing as mp

# from utils.vanillamcts import vanilla_mcts
from utils.utils import prepare_state_for_net, get_net_best_legal
from utils.chess_utils_local import action_to_move
from utils.networks import DemoNet
from utils.agent import Agent, Stockfish
from utils.self_play import SelfPlaySession
from utils.memory import ReplayMemory, Transition
from utils.evaluator import evaluator
from utils.agent import Agent
from utils.utils import Timer


# NOTE this file tests interaction with the PettingZoo Chess environment

import torch
from pettingzoo.classic import chess_v6
from tqdm import tqdm

def play_chess_game():
    env = chess_v6.env(render_mode=None)  # No GUI rendering
    env.reset()

    player_to_int = {"player_0": 1, "player_1": -1}
    move_bar = tqdm(range(1, 10000), desc="Playing Chess Moves", leave=True)

    print("\nMove | Current Player | Reward from env.last() | Reward from env.rewards[current_player] | Difference")
    print("-" * 100)

    while True:
        current_player = env.agent_selection  # Get the player whose turn it is
        observation, reward_last, termination, truncation, info = env.last()  # Reward from previous agent's action
        reward_current = env.rewards[current_player] # Reward stored for the current player
        other_player_reward = env.rewards[env.possible_agents[0] if env.agent_selection == env.possible_agents[1] else env.possible_agents[1]]

        # Print the difference between reward sources
        print(f"{current_player:14} | {reward_last:23} | {reward_current:41} | {reward_last - reward_current} | {other_player_reward}")

        # If the game is over, exit the loop
        if termination or truncation:
            if termination: print('Terminated')
            if truncation: print('Truncation')
            break

        # Make a random move (replace this with AI logic if needed)
        action_mask = observation['action_mask']
        action = env.action_space(current_player).sample(action_mask)
        env.step(action=action)

    print("\nGame Over!")


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

    agent_1 = Agent(
        version=0,
        network=net,
        sims=100
    )

    agent_2 = Agent(
        version=0,
        network=deepcopy(net),
        sims=100
    )

    # net.share_memory()
    times = []
    termination, truncation = False, False
    i = 0
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
            num_threads = 4

            if (i % 2) == 0:
                action, val = agent_1.inference(
                    board_state=deepcopy(env.board),
                    observation=observation['observation'],
                    device=device,
                    tau=1
                )

            elif (i % 2) == 1:
                action, val = agent_2.inference(
                    board_state=deepcopy(env.board),
                    observation=observation['observation'],
                    device=device,
                    tau=1
                )

            t = time.time()-start
            print(f'MCTS with {sims} sims completes after {t} s')
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
    print(f'Average MCTS (sims: {sims} threads: {num_threads}) time: {sum(times)/len(times)} s')

if __name__ == '__main__':
    from utils.training import Checkpoint
    cp = Checkpoint(compute_elo=False, verbose=True)
    memory = cp.download_from_blob("checkpoints/best_weights/replay_memory.pkl")
    print(memory[0].policy.max(), memory[331].policy.max(), memory[213].policy.max(), memory[393].policy.max())
