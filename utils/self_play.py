import numpy as np
from pettingzoo.classic import chess_v6
from utils.mcts import mcts  
from utils.memory import ReplayMemory
import torch
import torch.nn as nn
import random
from copy import deepcopy
from tqdm import tqdm


class SelfPlaySession:
    def __init__(
        self, 
        v_resign_start: float = -0.95, 
        disable_resignation_fraction: float = 0.1, 
        temperature_initial_moves: int = 30
    ):
        self.v_resign = v_resign_start
        self.disable_resignation_prob = disable_resignation_fraction
        self.false_positive_threshold = 0.05 # Keeping static
        self.full_game_resignations = 0
        self.false_resignations = 0
        self.total_resigned_games = 0
        self.temperature_initial_moves = temperature_initial_moves

    def should_resign(self, v: torch.Tensor) -> bool:
        return v < self.v_resign
    
    def should_disable_resignation(self) -> bool:
        return random.random() < self.disable_resignation_prob
    
    def adjust_v_resign(self) -> None:
        """Adjusts the resignation threshold based on false resignation rates."""
        if self.total_resigned_games == 0:
            return  # Avoid division by zero

        false_positive_rate = self.false_resignations / self.total_resigned_games

        if false_positive_rate > self.false_positive_threshold:
            # Reduce resignation frequency (raise v_resign, make it less aggressive)
            self.v_resign += 0.05  # Make resignations more cautious
            print(f"Increasing v_resign to {self.v_resign} to reduce false resignations.")
        elif false_positive_rate < self.false_positive_threshold / 2:
            # Allow slightly more aggressive resignations
            self.v_resign -= 0.05  
            print(f"Decreasing v_resign to {self.v_resign} to allow earlier resignations.")

        # Reset tracking counters for the next batch of games
        self.false_resignations = 0
        self.total_resigned_games = 0

    # TODO, make epsilon and alpha input params into run self play
    def run_self_play(
        self,
        training_data: ReplayMemory,
        network: nn.Module,
        device: torch.device,
        n_sims: int,
        num_games: int, 
        max_moves: int, 
    ) -> None:
        """
        Runs self-play using MCTS in PettingZoo's Chess environment.

        Args:
            training_data (ReplayMemory): Replay memory object.
            network: (nn.Module): Current best player agent_theta star.
            n_sims (int): Number of sims for mcts.
            num_games (int): Number of self-play games to generate.
            max_moves (int): Maximum number of moves per game before termination.
        """

        # env = chess_v6.env(render_mode='human')  
        env = chess_v6.env(render_mode=None)  
        player_to_int = {
            "player_0": 1,
            "player_1": -1
        }
        int_to_player = {
            1: "player_0",
            -1: 'player_1'
        }

        for game_idx in range(1, num_games+1):
            print('*'*50)
            print(f'Starting game #{game_idx}')
            should_disable = False
            supposed_winner = None # For resgin false positive logic
            env.reset()
            game_states = []
            move_policies = []
            players = []

            pbar = tqdm(range(1, max_moves+1))
            for move_idx in pbar:
                current_player = env.agent_selection
                observation, reward, termination, truncation, info = env.last()
                pbar.set_description(f"Running move {move_idx} for {current_player}")

                # Check for game termination
                if termination:
                    game_result = reward 
                    last_player = player_to_int[current_player]
                    winning_player = game_result * last_player
                    pbar.set_description(f"Game terminated for {current_player} at move {move_idx}. {winning_player} is the winner. Reward = {reward}, Last Player = {last_player}")
                    break

                if truncation:
                    last_player = player_to_int[current_player]
                    game_result = 0  
                    winning_player = 0
                    pbar.set_description(f"Game truncated for {current_player} at move {move_idx}. {winning_player} is the winner. Reward = {reward}, Last Player = {last_player}")
                    break

                state = observation['observation']
                tau = 1.0 if move_idx < self.temperature_initial_moves else 0

                # Run MCTS 
                pi, v, selected_move = mcts(deepcopy(env.board), net=network, device=device, tau=tau, sims=n_sims)  # NOTE pi should already be a probability distribution
                pi = pi.squeeze()
                # Store state, policy, and value
                game_states.append(torch.from_numpy(state.copy()))
                move_policies.append(pi)
                players.append(torch.tensor([player_to_int[current_player]]))

                # Resignation logic
                if move_idx > 10:
                    if self.should_resign(v=v):
                        self.total_resigned_games += 1
                        should_disable = self.should_disable_resignation()
                        supposed_winner = -player_to_int[current_player]
                        if should_disable:
                            self.full_game_resignations += 1
                        else:
                            print(f"Player {current_player} resigned at {move_idx}")
                            winning_player = supposed_winner
                            break

                # Play the move
                env.step(selected_move)

            else:
                game_result = 0  # Draw by reaching max moves
                winning_player = 0
                pbar.set_description("Game is a draw due to reached max moves.")

            # False resignation check
            if should_disable:
                is_false_positive = int(winning_player != supposed_winner)
                if is_false_positive:
                    self.false_resignations += 1

            # Assign game outcome to all stored states
            for state, policy, player in zip(game_states, move_policies, players):
                if player == winning_player:
                    adjusted_reward = 1
                elif winning_player == 0:
                    adjusted_reward = 0
                else:
                    adjusted_reward = -1

                training_data.push(state.float().permute(2, 0, 1), policy, torch.tensor([adjusted_reward], dtype=torch.float))  

            print(f"Completed game {game_idx}/{num_games}")

        # Adjust resignation threshold after batch of games
        self.adjust_v_resign()