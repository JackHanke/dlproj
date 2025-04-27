import torch
from tqdm import tqdm
from copy import deepcopy
import logging
from pettingzoo.classic import chess_v6

from utils.agent import Agent

logging.basicConfig(filename='dem0.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluator(
        challenger_agent: Agent, 
        current_best_agent: Agent, 
        device: torch.device,
        max_moves: int,
        num_games: int,
        checkpoint_client,  # The Checkpoint instance for saving state
        iteration: int,     # The current iteration number
        v_resign: float, 
        win_threshold: float = 0.55
    ) -> Agent:
    # watch external evaluation
    if current_best_agent.version == 'Stockfish':
        env = chess_v6.env(render_mode=None)  
    else:
        env = chess_v6.env(render_mode=None)  
        # env = chess_v6.env(render_mode=None)  

    # env = chess_v6.env()
    player_to_int = {"player_0": 1, "player_1": -1}

    challenger_agent_wins = 0
    draws = 0
    current_best_agent_wins = 0

    threshold_games = round(num_games * win_threshold)

    # Use a tqdm progress bar for the games
    for game_idx in range(1, num_games+1):
        if challenger_agent_wins >= threshold_games or current_best_agent_wins >= threshold_games:
            break
        
        logging.debug(f'Starting game #{game_idx}')
        env.reset()

        # Assign agents based on game index (alternate colors)
        if (game_idx % 2) == 0:
            player_to_agent = {
                "player_0": challenger_agent,
                "player_1": current_best_agent
            }
        else:
            player_to_agent = {
                "player_0": current_best_agent,
                "player_1": challenger_agent
            }

        logging.debug(f'{player_to_agent["player_0"].version} vs. {player_to_agent["player_1"].version}')
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
                logging.debug(f"Game terminated for {current_player} at move {move_idx}. {winning_player} is the winner. Reward = {reward}, Last Player = {last_player}, is truncated: {truncation}")
                break

            if truncation:
                winning_player = 0
                last_player = player_to_int[current_player]
                logging.debug(f"Game truncated for {current_player} at move {move_idx}. {winning_player} is the winner. Reward = {reward}, Last Player = {last_player}")
                break

            tau = 0  # No exploration during evaluation
            agent = player_to_agent[current_player]
            selected_move, v = agent.inference(board_state=deepcopy(env.board), starting_agent=current_player, device=device, tau=tau)

            # Resignation logic
            if move_idx > 10 and v < v_resign:
                winning_player = -1 * player_to_int[current_player]
                logging.debug(f"Player {current_player} resigned at move {move_idx} with value {v:.3f}")
                break

            env.step(selected_move)

        # If no termination or resignation occurred, consider the game a draw.
        if winning_player is None:
            winning_player = 0

        # Update win counts based on agent colors
        if (game_idx % 2) == 0:  # Challenger is white
            if winning_player == 1:
                challenger_agent_wins += 1
                logging.info("Challenger one.")
            elif winning_player == -1:
                current_best_agent_wins += 1
                logging.info("Current best one.")
            else:
                logging.info("Draw between current best and evaluator")
                draws += 1
        else:  # Challenger is black
            if winning_player == -1:
                challenger_agent_wins += 1
                logging.info("Challenger one.")
            elif winning_player == 1:
                current_best_agent_wins += 1
                logging.info("Current best one.")
            else:
                logging.info("Draw between current best and evaluator")
                draws += 1

        # Save evaluation state for resuming mid-evaluation
        eval_state = {
            'game_idx': game_idx,
            'challenger_agent_wins': challenger_agent_wins,
            'draws': draws,
            'current_best_agent_wins': current_best_agent_wins
        }
        checkpoint_client.save_file(
            obj=eval_state,
            blob_folder=f"checkpoints/iteration_{iteration}/evaluation",
            filename="evaluation_state.pkl"
        )

        logging.debug(f"Completed game {game_idx}/{num_games}")

        # --- Early‑stopping logic based on remaining games ----------------------
        # If the leading agent already has more wins than the trailing agent
        # can possibly achieve given the games left, stop the evaluation early.
        remaining_games = num_games - game_idx
        if challenger_agent_wins > current_best_agent_wins + remaining_games:
            logging.info(
                f"Early stopping: challenger leads {challenger_agent_wins}-{current_best_agent_wins} "
                f"with only {remaining_games} game(s) left — lead is mathematically unassailable."
            )
            break
        if current_best_agent_wins > challenger_agent_wins + remaining_games:
            logging.info(
                f"Early stopping: current best leads {current_best_agent_wins}-{challenger_agent_wins} "
                f"with only {remaining_games} game(s) left — lead is mathematically unassailable."
            )
            break
        # -----------------------------------------------------------------------

    win_percent = challenger_agent_wins/game_idx

    # if external evaluation, return number of games challenger agent won
    if current_best_agent.version == 'Stockfish':
        return challenger_agent_wins, draws, current_best_agent_wins, win_percent, game_idx

    logging.debug(f'Played {(game_idx + 1)} games. Challenger Agent points: {challenger_agent_wins}, Current Best Agent points: {current_best_agent_wins}')


    # else return best agent
    if challenger_agent_wins >= current_best_agent_wins:
        logging.info(f'Challenger agent v{challenger_agent.version} wins!')
        return_agent = challenger_agent  
    else:
        return_agent = current_best_agent

    return return_agent, challenger_agent_wins, draws, current_best_agent_wins, win_percent, game_idx

