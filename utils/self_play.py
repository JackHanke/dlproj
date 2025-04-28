import numpy as np
import torch
import torch.nn as nn
import random
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
    def __init__(
        self, 
        checkpoint_client: any,
        v_resign_start: float = -0.95, 
        disable_resignation_fraction: float = 0.1, 
        temperature_initial_moves: int = 30
    ):
        self.checkpoint_client = checkpoint_client
        self.v_resign = v_resign_start
        self.disable_resignation_prob = disable_resignation_fraction
        self.false_positive_threshold = 0.05
        self.full_game_resignations = 0
        self.false_resignations = 0
        self.total_resigned_games = 0
        self.temperature_initial_moves = temperature_initial_moves

    def should_resign(self, v: torch.Tensor) -> bool:
        return v < self.v_resign
    
    def should_disable_resignation(self) -> bool:
        return random.random() < self.disable_resignation_prob
    
    def adjust_v_resign(self) -> None:
        if self.total_resigned_games == 0:
            return
        false_positive_rate = self.false_resignations / self.total_resigned_games
        if false_positive_rate > self.false_positive_threshold:
            self.v_resign += 0.05
            logger.debug(f"Increasing v_resign to {self.v_resign} to reduce false resignations.")
        elif false_positive_rate < self.false_positive_threshold / 2:
            self.v_resign -= 0.05  
            logger.debug(f"Decreasing v_resign to {self.v_resign} to allow earlier resignations.")

        self.false_resignations = 0
        self.total_resigned_games = 0

    def run_self_play(
        self,
        iteration: int,
        transition_queue: torch.multiprocessing.Queue,
        network: nn.Module,
        device: torch.device,
        n_sims: int,
        num_games: int, 
        max_moves: int, 
        start_from_game_idx: int = 0
    ) -> None:
        env = chess_v6.env(render_mode=None)
        player_to_int = {"player_0": 1, "player_1": -1}

        for game_idx in range(1+start_from_game_idx, num_games+1):
            logger.debug(f'Starting game #{game_idx}')
            should_disable = False
            supposed_winner = None
            env.reset()
            game_states = []
            move_policies = []
            players = []

            move_bar = tqdm(range(1, max_moves+1))
            for move_idx in move_bar:
                current_player = env.agent_selection
                observation, reward, termination, truncation, info = env.last()
                move_bar.set_description(f"SelfPlay, Game {game_idx}: Move {move_idx} by Player {current_player}.")

                if termination:
                    last_player = player_to_int[current_player]
                    game_result = reward
                    winning_player = game_result * last_player
                    move_bar.set_description(f"Game terminated, winner is player {winning_player} with reward of {reward}")
                    break
                elif truncation:
                    last_player = player_to_int[current_player]
                    game_result = reward
                    winning_player = game_result * last_player
                    move_bar.set_description(f"Game truncated, winner is player {winning_player} with reward of {reward}")
                    break

                state = observation['observation']
                tau = 1.0 if move_idx < self.temperature_initial_moves else 0

                pi, v, selected_move, _ = mcts(
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
                players.append(torch.tensor([player_to_int[current_player]]))

                if move_idx > 10 and self.should_resign(v=v):
                    self.total_resigned_games += 1
                    should_disable = self.should_disable_resignation()
                    supposed_winner = -player_to_int[current_player]
                    if not should_disable:
                        winning_player = supposed_winner
                        break

                env.step(selected_move)

            else:
                winning_player = 0  # Draw

            if should_disable:
                if winning_player != supposed_winner:
                    self.false_resignations += 1

            # Push transitions from this game to queue
            for state, policy, player in zip(game_states, move_policies, players):
                if player == winning_player:
                    adjusted_reward = 1
                elif winning_player == 0:
                    adjusted_reward = 0
                else:
                    adjusted_reward = -1
                transition_queue.put((
                    state.float().permute(2, 0, 1).detach().cpu(),
                    policy.detach().cpu(),
                    torch.tensor([adjusted_reward], dtype=torch.float)
                ))


            logger.info(f"Pushed self-play data to memory after Game {game_idx}")
            self.checkpoint_client.save_log(iteration=iteration)
            self.checkpoint_client.save_file(obj=game_idx, blob_folder=f"checkpoints/iteration_{iteration}", filename='self_play_games_completed.pkl')
            logger.info(f"Saved self play game idx {game_idx} in iteration history.")
            logger.debug(f"Completed game {game_idx}/{num_games}")

        self.adjust_v_resign()