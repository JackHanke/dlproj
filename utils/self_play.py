import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import logging
import time
from copy import deepcopy
from tqdm import tqdm
from pettingzoo.classic import chess_v6
from utils.memory import ReplayMemory
from utils.mcts import mcts  

logger = logging.getLogger(__name__)
logging.basicConfig(filename='dem0.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.storage").setLevel(logging.WARNING)

class SelfPlaySession:
    def __init__(self, checkpoint_client: any, temperature_initial_moves: int = 30):
        self.checkpoint_client = checkpoint_client
        self.temperature_initial_moves = temperature_initial_moves

    def run_self_play(
        self,
        iteration: int,
        transition_queue: torch.multiprocessing.Queue,
        weight_queue: mp.Queue,
        network: nn.Module,
        device: torch.device,
        n_sims: int,
        num_games: int,
        max_moves: int,
        start_from_game_idx: int = 0
    ) -> None:
        env = chess_v6.env(render_mode=None)
        player_to_int = {"player_0": 1, "player_1": -1}

        # Load the most recent weights before starting games
        if weight_queue is not None:
            while not weight_queue.empty():
                latest_state = weight_queue.get()
                network.load_state_dict(latest_state)
                network.to(device)

        for game_idx in range(1 + start_from_game_idx, num_games + 1):
            logger.debug(f'Starting game #{game_idx}')
            env.reset()

            # Load the most recent weights before starting this game
            if weight_queue is not None:
                while not weight_queue.empty():
                    latest_state = weight_queue.get()
                    network.load_state_dict(latest_state)
                    network.to(device)

            game_states = []
            move_policies = []
            players = []

            move_bar = tqdm(range(1, max_moves + 1))
            for move_idx in move_bar:
                current_player = env.agent_selection
                observation, reward, termination, truncation, info = env.last()
                move_bar.set_description(f"SelfPlay, Game {game_idx}: Move {move_idx} by {current_player}")

                if termination or truncation:
                    last_player = player_to_int[current_player]
                    game_result = reward
                    winning_player = game_result * last_player
                    move_bar.set_description(f"Game ended, winner: {winning_player} (reward: {reward} for last player = {last_player})")
                    break

                state = observation['observation']
                tau = 1.0 if move_idx < self.temperature_initial_moves else 0

                # Non‑blocking weight refresh
                if weight_queue is not None and not weight_queue.empty():
                    latest_state = weight_queue.get()
                    network.load_state_dict(latest_state)
                    network.to(device)

                pi, _, selected_move, _ = mcts(
                    state=deepcopy(env.board),
                    starting_agent=current_player,
                    net=network,
                    device=device,
                    tau=tau,
                    sims=n_sims
                )
                pi = pi.squeeze()
                game_states.append(torch.from_numpy(state.copy()))
                move_policies.append(pi)
                players.append(player_to_int[current_player])

                env.step(selected_move)
            else:
                winning_player = 0  # Draw if no termination

            # Push transitions to queue
            for state, policy, player in zip(game_states, move_policies, players):
                if player == winning_player:
                    adjusted_reward = 1
                elif winning_player == 0:
                    adjusted_reward = 0
                elif player == -winning_player:
                    adjusted_reward = -1
                else:
                    raise ValueError("Invalid reward or player")

                transition_queue.put((
                    state.float().permute(2, 0, 1).detach().cpu(),
                    policy.detach().cpu(),
                    torch.tensor([adjusted_reward], dtype=torch.float)
                ))

            logger.info(f"Pushed self-play data after Game {game_idx}")
            self.checkpoint_client.save_log(iteration=iteration)
            self.checkpoint_client.save_file(obj=game_idx, blob_folder=f"checkpoints/iteration_{iteration}", filename='self_play_games_completed.pkl')
            logger.debug(f"Completed game {game_idx}/{num_games}")