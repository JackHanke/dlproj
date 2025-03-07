import torch
import numpy as np
from pettingzoo.classic import chess_v6
import timeit
from contextlib import ContextDecorator


class Timer(ContextDecorator):
    def __enter__(self):
        self.start_time = timeit.default_timer()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = timeit.default_timer()
        elapsed_time = self.end_time - self.start_time
        print(f"Execution time: {elapsed_time:.6f} seconds")

# random argmax for RL action choices
def rand_argmax(tens):
    max_inds, _ = torch.where(tens == tens.max())
    return np.random.choice(max_inds)

def get_action_mask_from_state(state: np.ndarray, player: any) -> np.ndarray:
    """
    Returns the action mask for that player given a state and player.

    Returns:
        action_mask (np.ndarray): A binary array where 1 = legal move, 0 = illegal move.
    """
    env = chess_v6.env() 
    env.reset()
    env.state = state # Set the board to the given state
    
    env.agent_selection = player  # Set the current player
    observation, _, _, _, _ = env.last()  # Get the observation

    action_mask = observation["action_mask"]  
    return action_mask

# takes raw policy logits and returns legal move logits
# prepare PettingZoo env state to network-ready state
def prepare_state_for_net(state):
    return torch.tensor(state).float().permute(2, 0, 1).unsqueeze(0)

# transform raw network output into probability distribution over legal moves the size of the all-action vector
def renormalize_network_output(policy_vec, legal_moves):
    policy = torch.squeeze(policy_vec)
    # TODO why is there not a better way for this
    logits = torch.mul(policy, legal_moves) -20*(1-legal_moves)
    renormalized_vec = torch.nn.functional.softmax(logits, dim=0)
    return renormalized_vec

def filter_legal_moves(policy_vec, legal_moves):
    policy = torch.squeeze(policy_vec)
    # get legal move indices
    # legal_moves = torch.tensor(legal_moves)
    legal_indices = torch.nonzero(legal_moves)
    return policy[legal_indices]

# 
def filter_legal_moves_and_renomalize(policy_vec, legal_moves):
    policy = torch.squeeze(policy_vec)
    # get legal move indices
    legal_indices = torch.nonzero(legal_moves)
    legal_logits = policy[legal_indices]
    p_vec = torch.nn.functional.softmax(legal_logits, dim=0)
    return p_vec



# get index of highest logit among legal positions
def get_net_best_legal(policy_vec, legal_moves):
    legal_logits = filter_legal_moves(policy_vec=policy_vec, legal_moves=legal_moves)
    # get maximum value 
    max_val = torch.max(legal_logits)
    # get action that has that maximum value among legal moves
    action = (policy_vec.squeeze(0) == max_val).nonzero(as_tuple=True)[0].item()
    return action
