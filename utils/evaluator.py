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
    checkpoint_client,
    iteration: int,
    win_threshold: float = 0.55
) -> Agent:

    env = chess_v6.env(render_mode=None)
    player_to_int = {"player_0": 1, "player_1": -1}

    challenger_agent_wins = 0
    draws = 0
    current_best_agent_wins = 0
    start_from_game = 1

    challenger_is_white = True if start_from_game % 2 == 0 else False

    # Determine evaluation state path based on agent version
    if current_best_agent.version == 'Stockfish':
        eval_state_blob = f"checkpoints/iteration_{iteration}/stockfish_evaluation/stockfish_evaluation_state.pkl"
    else:
        eval_state_blob = f"checkpoints/iteration_{iteration}/evaluation/evaluation_state.pkl"

    # 🔥 Try loading cached evaluation state
    if checkpoint_client.blob_exists(eval_state_blob):
        try:
            eval_state = checkpoint_client.download_from_blob(eval_state_blob)
            challenger_agent_wins = eval_state['challenger_agent_wins']
            draws = eval_state['draws']
            current_best_agent_wins = eval_state['current_best_agent_wins']
            start_from_game = eval_state['game_idx'] + 1  # start from the next game
            logging.info(f"✅ Resuming evaluation from Game {start_from_game}. Previous wins: {challenger_agent_wins}, Draws: {draws}, Losses: {current_best_agent_wins}")
        except Exception as e:
            logging.warning(f"⚠️ Failed to load evaluation state: {e}. Starting fresh.")
    else:
        logging.info(f"🆕 No previous evaluation found. Starting fresh.")

    threshold_games = round(num_games * win_threshold)

    for game_idx in range(start_from_game, num_games + 1):
        if challenger_agent_wins >= threshold_games or current_best_agent_wins >= threshold_games:
            break

        logging.info(f'Starting game #{game_idx}')
        env.reset()

        if challenger_is_white:
            player_to_agent = {
                "player_0": challenger_agent,
                "player_1": current_best_agent
            }
        else:
            player_to_agent = {
                "player_0": current_best_agent,
                "player_1": challenger_agent
            }

        challenger_is_white = not challenger_is_white

        winning_player = None

        move_bar = tqdm(range(1, max_moves + 1), desc=f"Game {game_idx} moves", leave=True)
        for move_idx in move_bar:
            current_player = env.agent_selection
            move_bar.set_description(f"Evaluator, Game {game_idx}: Move {move_idx} by {current_player}.")
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                game_result = reward
                last_player = player_to_int[current_player]
                winning_player = game_result * last_player
                break

            tau = 0
            agent = player_to_agent[current_player]
            selected_move, v = agent.inference(board_state=deepcopy(env.board), starting_agent=current_player, device=device, tau=tau)

            env.step(selected_move)

        if winning_player is None:
            winning_player = 0

        if (game_idx % 2) == 0:
            if winning_player == 1:
                challenger_agent_wins += 1
                logging.info("Challenger one!")
            elif winning_player == -1:
                current_best_agent_wins += 1
                logging.info("Current best one.")
            else:
                logging.info('Draw')
                draws += 1
        else:
            if winning_player == -1:
                challenger_agent_wins += 1
                logging.info("Challenger one!")
            elif winning_player == 1:
                current_best_agent_wins += 1
                logging.info("Current best one.")
            else:
                logging.info('Draw')
                draws += 1

        # 🔥 Save after every game
        eval_state = {
            'game_idx': game_idx,
            'challenger_agent_wins': challenger_agent_wins,
            'draws': draws,
            'current_best_agent_wins': current_best_agent_wins
        }
        # Save to appropriate path based on agent version
        if current_best_agent.version == 'Stockfish':
            checkpoint_client.save_file(
                obj=eval_state,
                blob_folder=f"checkpoints/iteration_{iteration}/stockfish_evaluation",
                filename="stockfish_evaluation_state.pkl"
            )
        else:
            checkpoint_client.save_file(
                obj=eval_state,
                blob_folder=f"checkpoints/iteration_{iteration}/evaluation",
                filename="evaluation_state.pkl"
            )

        logging.info(f"Completed game {game_idx}/{num_games}")

        # --- Early‑stopping logic ----------------------
        remaining_games = num_games - game_idx
        if challenger_agent_wins > current_best_agent_wins + remaining_games:
            logging.info("Early stopping: Challenger guaranteed win.")
            break
        if current_best_agent_wins > challenger_agent_wins + remaining_games:
            logging.info("Early stopping: Current best guaranteed win.")
            break
        # ------------------------------------------------

    total_games_played = challenger_agent_wins + current_best_agent_wins + draws
    win_percent = challenger_agent_wins / total_games_played

    # For Stockfish evaluation return stats
    if current_best_agent.version == 'Stockfish':
        return challenger_agent_wins, draws, current_best_agent_wins, win_percent, num_games

    logging.info(f'Finished {num_games} games. Challenger: {challenger_agent_wins} wins, Current Best: {current_best_agent_wins} wins.')

    # Otherwise return the better agent
    if challenger_agent_wins >= current_best_agent_wins:
        logging.info(f"✅ Challenger v{challenger_agent.version} is the new best agent!")
        return_agent = challenger_agent
    else:
        return_agent = current_best_agent

    return return_agent, challenger_agent_wins, draws, current_best_agent_wins, win_percent, num_games
