import numpy as np
from pettingzoo.classic import chess_v6
from utils.mcts import mcts  
from utils.utils import get_action_mask_from_state
from utils.memory import ReplayMemory
import torch
import torch.nn as nn
import random
from copy import deepcopy


# TODO, make epsilon and alpha input params into run self play
class SelfPlayAgent:
    def __init__(self, v_resign_start: float = -0.95):
        self.v_resign = v_resign_start
        self.disable_resignation_prob = 0.1
        self.false_positive_threshold = 0.05
        self.full_game_resignations = 0
        self.false_resignations = 0
        self.total_resigned_games = 0

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

    def run_self_play(
        self,
        training_data: ReplayMemory,
        network: nn.Module,
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

        env = chess_v6.env()  
        player_to_int = {
            "player_0": 1,
            "player_1": -1
        }

        for game_idx in range(num_games):
            print(f'Starting game #{game_idx}')
            should_disable = False
            supposed_winner = None # For resgin false positive logic
            env.reset()
            game_states = []
            move_policies = []
            players = []

            for move_idx in range(max_moves):
                current_player = env.agent_selection
                observation, reward, termination, truncation, info = env.last()

                # Check for game termination
                if termination:
                    print(f"Game terminated for {current_player} at move {move_idx}")
                    game_result = env.rewards[current_player]  
                    last_player = player_to_int[current_player]
                    winning_player = game_result * last_player
                    break

                if truncation:
                    print(f"Game truncated for {current_player} at move {move_idx}")
                    game_result = 0  
                    winning_player = 0
                    break

                state = observation['observation']
                tau = 1.0 if move_idx < 30 else 0

                # Run MCTS 
                print('Starting mcts...')
                pi, v, selected_move = mcts(deepcopy(env.board), net=network, tau=tau, sims=n_sims)  # NOTE pi should already be a probability distribution
                print('Finished mcts!')
                print(f"Selected move: {selected_move}")
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

                training_data.push(state, policy, adjusted_reward)  

            print(f"Completed game {game_idx + 1}/{num_games}")

        # Adjust resignation threshold after batch of games
        self.adjust_v_resign()