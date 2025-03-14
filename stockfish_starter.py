import torch
import logging
import time
import pickle
from tqdm import tqdm 
from copy import deepcopy
from utils.agent import Agent, Stockfish
from utils.memory import ReplayMemory
from pettingzoo.classic import chess_v6

logger = logging.getLogger(__name__)

def stockfish_starter(
        challenger_agent: Agent, 
        current_best_agent: Agent, 
        device: torch.device,
        max_moves: int,
        num_games: int
    ):

    # env = chess_v6.env(render_mode='human') 
    env = chess_v6.env(render_mode=None) 

    player_to_int = {"player_0": 1, "player_1": -1}

    challenger_agent_wins = 0
    draws = 0
    current_best_agent_wins = 0
    training_data = []

    # Use a tqdm progress bar for the games
    for game_idx in range(1, num_games+1):

        game_states = []
        move_policies = []
        players = []

        logger.debug(f'Starting game #{game_idx}')
        env.reset()

        # Assign agents based on game index (alternate colors)
        player_to_agent = {
            "player_0": challenger_agent,
            "player_1": current_best_agent
        }

        logger.debug(f'{player_to_agent["player_0"].version} vs. {player_to_agent["player_1"].version}')
        # print(f'{player_to_agent["player_0"].version} vs. {player_to_agent["player_1"].version}')

        winning_player = None

        # Use a tqdm progress bar for moves within the game
        move_bar = tqdm(range(1, max_moves + 1), desc=f"Game {game_idx} moves", leave=True)
        for move_idx in move_bar:
            current_player = env.agent_selection
            move_bar.set_description(f"Evaluator, Game {game_idx}: Move {move_idx} by Player {current_player}.")
            observation, reward, termination, truncation, info = env.last()

            # Check for game termination
            if termination:
                # env.rewards[current_player] is 1 if the current player just won, -1 if lost, 0 if draw
                game_result = reward
                last_player = player_to_int[current_player]
                winning_player = game_result * last_player
                logger.debug(f"Game terminated for {current_player} at move {move_idx}. {winning_player} is the winner. Reward = {reward}, Last Player = {last_player}, is truncated: {truncation}")
                break

            if truncation:
                winning_player = 0
                last_player = player_to_int[current_player]
                logger.debug(f"Game truncated for {current_player} at move {move_idx}. {winning_player} is the winner. Reward = {reward}, Last Player = {last_player}")
                break

            state = observation['observation']

            tau = 0  # No exploration during evaluation
            agent = player_to_agent[current_player]
            selected_move, v = agent.inference(
                board_state=deepcopy(env.board), 
                observation=observation['observation'], 
                device=device, 
                tau=tau
            )

            pi = [0 for _ in range(4672)]
            pi[selected_move] = 1
            pi = torch.tensor(pi)

            # Store state, policy, and value
            game_states.append(torch.from_numpy(state.copy()))
            move_policies.append(pi)
            players.append(torch.tensor([player_to_int[current_player]]))


            env.step(selected_move)

        # If no termination or resignation occurred, consider the game a draw.
        if winning_player is None:
            winning_player = 0

        # Update win counts based on agent colors
        if (game_idx % 2) == 0:  # Challenger is white
            if winning_player == 1:
                challenger_agent_wins += 1
            elif winning_player == -1:
                current_best_agent_wins += 1
            else:
                draws += 1
        else:  # Challenger is black
            if winning_player == -1:
                challenger_agent_wins += 1
            elif winning_player == 1:
                current_best_agent_wins += 1
            else:
                draws += 1

        for state, policy, player in zip(game_states, move_policies, players):
            single_push_start_time = time.time()
            if player == winning_player:
                adjusted_reward = 1
            elif winning_player == 0:
                adjusted_reward = 0
            else:
                adjusted_reward = -1

            training_data.append(
                (
                    state.float().permute(2, 0, 1), 
                    policy, 
                    torch.tensor([adjusted_reward], dtype=torch.float),
                    player,
                    winning_player
                )
            )  

        print(f"Completed game {game_idx}/{num_games}")
        # logger.debug(f"Completed game {game_idx}/{num_games}")

        if (game_idx % 5) == 0:
            with open(f'tests/stock_data.pkl', 'wb') as f:
                pickle.dump(training_data, f)


if __name__ == '__main__':
    stockfish_level = 0
    stockfish_0 = Stockfish(level=stockfish_level, move_time=0.2)
    stockfish_1 = Stockfish(level=stockfish_level, move_time=0.2)

    start = time.time()
    stockfish_starter(
        challenger_agent = stockfish_0, 
        current_best_agent = stockfish_1, 
        device = 'cpu',
        max_moves = 250,
        num_games = 9999 
    ) # ahhh

    with open(f'tests/stock_data.pkl', 'rb') as f:
        training_data = pickle.load(f)

    print(f'Created {len(training_data)} datapoints for training in {time.time()-start} s')
