import torch
import time
import json
import pickle
from tqdm import tqdm 
import gc
from copy import deepcopy
from pettingzoo.classic import chess_v6
import numpy as np

from utils.chess_utils_local import get_observation, action_to_move
from utils.agent import Agent, Stockfish
from utils.memory import ReplayMemory


def stockfish_starter(
        challenger_agent: Stockfish, 
        current_best_agent: Stockfish, 
        device: torch.device,
        max_moves: int,
        num_games: int
    ):

    # env = chess_v6.env(render_mode='human') 
    env = chess_v6.env(render_mode=None) 

    player_to_int = {"player_0": 1, "player_1": -1}

    training_data = []

    books = 0
    datapoints = 0

    # Use a tqdm progress bar for the games
    for game_idx in range(1, num_games+1):
        board_history = np.zeros((8, 8, 104), dtype=bool)


        if len(training_data) >= 100000:
            training_data = [] # flush data
            gc.collect()
            books +=1 #increment pickle file

        game_states = []
        move_policies = []
        players = []
        player_str_list = []
        board_history_list = []

        env.reset()

        # Assign agents based on game index (alternate colors)
        player_to_agent = {
            "player_0": challenger_agent,
            "player_1": current_best_agent
        }

        winning_player = None

        # Use a tqdm progress bar for moves within the game
        # move_bar = tqdm(range(1, max_moves + 1), desc=f"Game {game_idx} moves", leave=True)
        # for move_idx in move_bar:
        for move_idx in range(1, max_moves + 1):
            current_player = deepcopy(env.agent_selection)
            current_player_idx = env.agents.index(deepcopy(env.agent_selection))
            # move_bar.set_description(f"Game {game_idx}, Move {move_idx}")
            observation, reward, termination, truncation, info = env.last()
            datapoints += 1
            # Check for game termination
            if termination:
                # env.rewards[current_player] is 1 if the current player just won, -1 if lost, 0 if draw
                game_result = reward
                last_player = player_to_int[current_player]
                winning_player = game_result * last_player
                break

            if truncation:
                winning_player = 0
                last_player = player_to_int[current_player]
                break

            # state = observation['observation']
            state = deepcopy(env.board)

            tau = 0  # No exploration during evaluation
            agent = player_to_agent[current_player]
            selected_move, v = agent.inference(
                board_state=deepcopy(env.board), 
                observation=None, 
                device=device, 
                tau=tau
            )

            # pi = [0 for _ in range(4672)]
            # pi[selected_move] = 1
            pi = selected_move
            # pi = torch.tensor(pi)

            # Store state, policy, and value
            game_states.append(state)
            move_policies.append(pi)
            players.append(torch.tensor([player_to_int[current_player]]))
            player_str_list.append(current_player)
            board_history_list.append(board_history)

            env.step(selected_move)
            
            move = action_to_move(board=state, action=selected_move, player=current_player_idx)
            state.push(move)

            next_board = get_observation(state, player=0)
            board_history = np.dstack(
                (next_board[:, :, 7:], board_history[:, :, :-13])
            )

        

        # If no termination or resignation occurred, consider the game a draw.
        if winning_player is None:
            winning_player = 0

        for state, policy, player, player_string, bh in zip(game_states, move_policies, players, player_str_list, board_history_list):
            if player == winning_player:
                adjusted_reward = 1.0
            elif winning_player == 0:
                adjusted_reward = 0.0
            else:
                adjusted_reward = -1.0

            training_data.append(
                (
                    state, 
                    policy, 
                    adjusted_reward,
                    player_string,
                    bh
                )
            )  

        if (game_idx % 5) == 0:
            print(f'Number of datapoints created: {datapoints}')
            with open(f'data/moves-str-{str(books)}.pkl', 'wb') as f:
                pickle.dump(training_data, f)
            print(f'Done writing out!')

if __name__ == '__main__':
    stockfish_level = 0
    stockfish_0 = Stockfish(level=stockfish_level, move_time=0.1)
    stockfish_1 = Stockfish(level=stockfish_level, move_time=0.1)

    start = time.time()
    stockfish_starter(
        challenger_agent = stockfish_0, 
        current_best_agent = stockfish_1, 
        device = 'cpu',
        max_moves = 250,
        num_games = 9999
    ) # ahhh
    full = time.time()-start

    with open(f'data/moves-str-0.pkl', 'rb') as f:
        training_data = pickle.load(f)
        print(f'Created {len(training_data)} datapoints for training in {full} s, {len(training_data)/full} data per sec')
