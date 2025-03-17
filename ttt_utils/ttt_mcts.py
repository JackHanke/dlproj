import torch
from torch.distributions import Categorical
import time
import chess
import threading
import logging
import numpy as np
import concurrent.futures
import gc
from copy import deepcopy

from utils.chess_utils_local import get_observation, legal_moves, result_to_int, get_action_mask, actions_to_moves, moves_to_actions
from utils.utils import prepare_state_for_net, filter_legal_moves, filter_legal_moves_and_renomalize, renormalize_network_output, rand_argmax

# NOTE This implementation mirrors all black boards to be consistent with how the PettingZoo source interacts with python-chess
# NOTE this code refers to a chess.Move and a UCI string as a 'move' while the action indexes provided by Pettingzoo as an 'action'
# NOTE virtual loss is for high number of threads searching the same tree. currently is not implemented
VIRTUAL_LOSS = 1

# NOTE edited from source. I hate PettinZoo so fucking much!
def observe(board, agent_index):
    board_vals = np.array(board.squares).reshape(3, 3)
    cur_player = agent_index
    opp_player = (cur_player + 1) % 2

    cur_p_board = np.equal(board_vals, cur_player + 1)
    opp_p_board = np.equal(board_vals, opp_player + 1)

    observation = np.stack([cur_p_board, opp_p_board], axis=2).astype(np.int8)
    # legal_moves = board._legal_moves()
    legal_moves = [i for i, mark in enumerate(board.squares) if mark == 0]

    action_mask = np.zeros(9, "int8")
    for i in legal_moves:
        action_mask[i] = 1

    return {"observation": observation, "action_mask": action_mask}

# node of search tree
class Node:
    def __init__(
            self, 
            state: chess.Board, 
            observation_tensor: torch.tensor,
            parent = None, 
            prior: float = 0.0
        ):
        self.state = state # python-chess board object
        self.observation_tensor = observation_tensor # PettingZoo friendly representation, tracked for board history
        self.parent = parent # parent node of Node, None if root Node
        self.children = {} # dictionary of {chess.Move: child_Node}
        self.n = 0 # number of times Node has been visited
        self.w = 0 # total reward from this Node
        self.q = 0.0 # mean reward from this Node
        self.virtual_loss = 0 # virtual loss for node
        self.p = prior # prior from network

# creates the final pi tensor (same size as network output)
def create_pi_vector(node: Node, tau: float):
    pi = [0 for _ in range(9)]

    if tau == 0: # if tau ==0, find action with highest visit count, create 1-hot pi
        highest_n, most_visited_action = 0, None
        for action, child in node.children.items():
            if child.n > highest_n:
                highest_n = child.n
                most_visited_action = action
        pi[most_visited_action] = 1
        pi = torch.tensor(pi)
        return pi

    # else construct a distribution including illegal moves with zero probability
    pi_denom = sum([child.n**(1/tau) for child in node.children.values()])
    for action, child in node.children.items():
        val = (child.n ** (1/tau))/pi_denom
        pi[action] = val
    pi = torch.tensor(pi)
    return pi

# backup function for MCTS, loops until hitting the root node (which is the node without a parent)
def backup(node: Node, v: float = 0.0):
    with threading.Lock():
        while node is not None:
            node.n += 1
            node.w += v
            node.q = node.w / node.n
            v = -v  
            node = node.parent


@torch.no_grad()
def simulate(
        node: Node, 
        net: torch.nn.Module, 
        device: torch.device = None, 
        recursion_count: int = 0, 
        verbose: bool = False,
        c_puct: int = 3,
    ):
    
    legal_moves = [i for i, mark in enumerate(node.state.squares) if mark == 0]

    # a) Selection - Traverse down the tree using UCB
    if len(node.children) > 0:
        best_score = -float('inf')
        selected_move = None
        n = node.n + 1e-6  # Avoid division by zero

        for move in legal_moves:
            child = node.children[move]
            q = child.q
            n_val = child.n
            p = child.p
            u = c_puct * np.sqrt(n) * p / (1 + n_val)  # UCB calculation
            score = q + u
            if score > best_score:
                best_score = score
                selected_move = move

        next_node = node.children[selected_move]
        return simulate(next_node, net, device, recursion_count+1, verbose, c_puct)

    # b) Expansion - If at a leaf node, check if it's terminal or create children
    if node.state.check_game_over():
        # v = result_to_int(node.state.result(claim_draw=True))  # Terminal value
        winner = node.state.check_for_winner()
        if winner == -1: v = 0
        elif winner == 1: v = 1
        elif winner == 2: v = -1
        return backup(node, v)

    # c) Evaluation - Run network inference to get prior and value
    agent_index = int((len([i for i in node.state.squares if i == 0]) % 2) == 1) # if there are an odd number of pieces, player 1's turnj
    new_observation = observe(board=node.state, agent_index=agent_index)
    state_tensor = prepare_state_for_net(new_observation['observation'].copy()).to(device)
    
    policy, net_v = net(state_tensor)
    action_mask = torch.tensor(new_observation['action_mask'])
    p_vec = filter_legal_moves_and_renomalize(policy, action_mask)

    if recursion_count == 0:  # Add Dirichlet noise at root
        epsilon = 0.25
        alpha = 0.03
        num_legal = len(p_vec)
        noise = torch.distributions.Dirichlet(torch.full(p_vec.shape, alpha)).sample().to(device)
        p_vec = ((1 - epsilon) * p_vec) + (epsilon * noise)

    for move, p_val in zip(legal_moves, p_vec):
        next_state = deepcopy(node.state)
        next_state.play_turn(agent=agent_index, pos=move)
        new_node = Node(
            state=next_state,
            observation_tensor=None,
            parent=node,
            prior=p_val.item()
        )
        node.children[move] = new_node

    # d) Backup - Propagate the value estimate up the tree
    return backup(node, net_v)


# conduct Monte Carlo Tree Search for sims sims and the python-chess state
# using network net and temperature tau
@torch.no_grad()
def mcts(
        state: chess.Board, 
        observation: torch.tensor,
        net: torch.nn.Module, 
        tau: int, 
        node: Node = None,
        c_puct: int = 3,
        sims: int = 15, 
        num_threads: int = 1,
        device: torch.device = None, 
        verbose: bool = False,
        inference_mode: bool = False
    ):
        
    root = Node(state=state, observation_tensor=observation)

    # expand tree with num_threads threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(simulate, root, net, device, 0, verbose, c_puct) for _ in range(sims)]
        _ = [f.result() for f in futures]

    # construct pi from root node and given tau value
    pi = create_pi_vector(node=root, tau=tau)
    # value is the mean value from the root
    value = root.q
    # create pi distribution
    m = Categorical(pi)
    # if inference, just pick the best move
    if inference_mode: sampled_action = int(pi.argmax(-1).item())
    # if not, sample from pi
    else: sampled_action = int(m.sample().item())

    sub_tree = None
    root = None # dereference root for memory management
    gc.collect() # collect garbage NOTE this does nothing substantial

    return pi, value, sampled_action, sub_tree
