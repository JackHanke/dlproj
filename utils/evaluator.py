import torch
from tqdm import tqdm
from copy import deepcopy
import logging
from pettingzoo.classic import chess_v6

from utils.agent import Agent

logger = logging.getLogger(__name__)

def evaluator(
        challenger_agent: Agent, 
        current_best_agent: Agent, 
        device: torch.device,
        max_moves: int,
        num_games: int,
        v_resign: float, 
        win_threshold: float = 0.55
    ) -> Agent:
    env = chess_v6.env(render_mode='human')  
    # env = chess_v6.env()
    player_to_int = {"player_0": 1, "player_1": -1}

    challenger_agent_wins = 0
    current_best_agent_wins = 0

    threshold_games = round(num_games * win_threshold)

    # Use a tqdm progress bar for the games
    for game_idx in range(1, num_games+1):
        if challenger_agent_wins >= threshold_games or current_best_agent_wins >= threshold_games:
            break
        
        logger.debug(f'Starting game #{game_idx}')
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

        winning_player = None

        # Use a tqdm progress bar for moves within the game
        move_bar = tqdm(range(1, max_moves + 1), desc=f"Game {game_idx} moves", leave=False)
        for move_idx in move_bar:
            move_bar.set_description(f"Move {move_idx} - {env.agent_selection}")
            current_player = env.agent_selection
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

            tau = 0  # No exploration during evaluation
            agent = player_to_agent[current_player]
            selected_move, v = agent.inference(board_state=deepcopy(env.board), observation=observation['observation'], device=device, tau=tau)

            # Resignation logic
            if move_idx > 10 and v < v_resign:
                winning_player = -1 * player_to_int[current_player]
                logger.debug(f"Player {current_player} resigned at move {move_idx} with value {v:.3f}")
                break

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
                challenger_agent_wins += 0.5
                current_best_agent_wins += 0.5
        else:  # Challenger is black
            if winning_player == -1:
                challenger_agent_wins += 1
            elif winning_player == 1:
                current_best_agent_wins += 1
            else:
                challenger_agent_wins += 0.5
                current_best_agent_wins += 0.5

        logger.debug(f"Completed game {game_idx}/{num_games}")

    logging.info(f'Played {(game_idx + 1)} games. Challenger Agent points: {challenger_agent_wins}, Current Best Agent points: {current_best_agent_wins}')

    # if external evaluation, return number of games challenger agent won
    if current_best_agent.version == 'Stockfish':
        return (challenger_agent_wins/(game_idx + 1)), (current_best_agent_wins/(game_idx + 1))
    # else return best agent
    return challenger_agent if challenger_agent_wins >= current_best_agent_wins else current_best_agent

