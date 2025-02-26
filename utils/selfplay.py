import numpy as np
from pettingzoo.classic import chess_v6
from utils.mcts import mcts  
from utils.utils import get_action_mask_from_state
from utils.memory import ReplayMemory
import torch
import torch.nn as nn


# TODO implement resignation policy
def self_play(
    network: nn.Module,
    n_sims: int,
    num_games: int, 
    max_moves: int, 
    max_replay_len: int, 
    epsilon: float = 0.25, 
    alpha: float = 0.03, 
) -> ReplayMemory:
    """
    Runs self-play using MCTS in PettingZoo's Chess environment.

    Args:
        num_games (int): Number of self-play games to generate.
        max_moves (int): Maximum number of moves per game before termination.
        max_replay_len (int): Max capacity for Replay Memory.
        epsilon (float): Controls noise level.
        alpha (float): Dirchlet noise.

    Returns:
        ReplayMemory containing tuples (state, policy, game_result) for training.
    """

    env = chess_v6.env()  
    training_data = ReplayMemory(maxlen=max_replay_len)
    player_to_int = {
        "player_0": 1,
        "player_1": -1
    }

    for game_idx in range(num_games):
        env.reset()
        game_states = []
        move_policies = []
        players = []

        for move_idx in range(max_moves):
            current_player = env.agent_selection
            state = env.state()
            tau = 1.0 if move_idx < 30 else 1e-5  

            # Run MCTS 
            pi, _ = mcts(state, net=network, tau=tau, sims=n_sims)  # NOTE pi should already be a probability distribution

            # Store state, policy, and value
            game_states.append(state)
            move_policies.append(pi)
            players.append(player_to_int[current_player])

            # Apply Dirchlet noise only at root
            if move_idx == 0:
                dirchlet_noise = np.random.dirichlet([alpha] * len(pi))
                pi = (1-epsilon) * pi + epsilon * dirchlet_noise

            # Get action mask and select move
            action_mask = get_action_mask_from_state(state=state, player=current_player)

            # Ensure action mask is boolean
            action_mask = np.array(action_mask, dtype=bool)

            # Apply temperature annealing
            legal_moves_p = np.power(pi[action_mask], 1 / tau)
            legal_moves_p /= np.sum(legal_moves_p)  

            # Get valid move indices
            legal_moves_idx = np.where(action_mask)[0]

            # Select move based on MCTS policy
            if tau > 1e-5:  
                selected_move = np.random.choice(legal_moves_idx, p=legal_moves_p)  
            else:
                selected_move = legal_moves_idx[np.argmax(pi[action_mask])]  

            # Play the move
            env.step(selected_move)

            # Check for game termination
            if env.terminations[current_player]:
                game_result = env.rewards[current_player]  
                last_player = player_to_int[current_player]
                winning_player = game_result * last_player # Sanity check p*r=wp: (1*1=1, 1*-1=-1, -1*1=-1, -1*-1=1)
                break
        else:
            game_result = 0  # Draw by reaching max moves
            winning_player = 0

        # Assign game outcome to all stored states
        for state, policy, player in zip(game_states, move_policies, players):
            if player == winning_player:
                adjusted_reward = 1  # This player won
            elif winning_player == 0:
                adjusted_reward = 0  # Draw
            else:
                adjusted_reward = -1  # This player lost

            training_data.push(state, policy, adjusted_reward)  

        print(f"Completed game {game_idx + 1}/{num_games}")

    return training_data

