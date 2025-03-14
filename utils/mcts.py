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
    pi = [0 for _ in range(4672)]

    if tau == 0: # if tau ==0, find action with highest visit count, create 1-hot pi
        highest_n, most_visited_action = 0, None
        for move, child in node.children.items():
            action = moves_to_actions[move]
            if child.n > highest_n:
                highest_n = child.n
                most_visited_action = action
        pi[most_visited_action] = 1
        pi = torch.tensor(pi)
        return pi

    # else construct a distribution including illegal moves with zero probability
    pi_denom = sum([child.n**(1/tau) for child in node.children.values()])
    for move, child in node.children.items():
        action = moves_to_actions[move]
        val = (child.n ** (1/tau))/pi_denom
        pi[action] = val
    pi = torch.tensor(pi)
    return pi

# compute if a board is in a terminal state (ripped from PettingZoo source)
def is_game_over(board: chess.Board):
    # if game ends
    next_legal_moves = legal_moves(board)
    is_stale_or_checkmate = not any(next_legal_moves)
    # claim draw is set to be true to align with normal tournament rules
    is_insufficient_material = board.is_insufficient_material()
    can_claim_draw = board.can_claim_draw()
    game_over = can_claim_draw or is_stale_or_checkmate or is_insufficient_material
    return game_over

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
    """
    Simulation step of MCTS following AlphaZero structure:
    a) Selection - Traverse based on UCB
    b) Expansion - If new node, add children
    c) Evaluation - Use NN to get prior & value estimate
    d) Backup - Propagate values back
    """
    
    # a) Selection - Traverse down the tree using UCB
    if len(node.children) > 0:
        best_score = -float('inf')
        selected_move = None
        n = node.n + 1e-6  # Avoid division by zero

        for move in node.state.legal_moves:
            child = node.children[move.uci()]
            q = child.q
            n_val = child.n
            p = child.p
            u = c_puct * np.sqrt(n) * p / (1 + n_val)  # UCB calculation
            score = q + u
            if score > best_score:
                best_score = score
                selected_move = move

        next_node = node.children[selected_move.uci()]
        return simulate(next_node, net, device, recursion_count+1, verbose, c_puct)

    # b) Expansion - If at a leaf node, check if it's terminal or create children
    if is_game_over(node.state):
        v = result_to_int(node.state.result(claim_draw=True))  # Terminal value
        return backup(node, v)

    # c) Evaluation - Run network inference to get prior and value
    next_board = get_observation(node.state, player=0)  
    new_observation = np.dstack((next_board[:, :, 7:], node.observation_tensor[:, :, :-13]))
    state_tensor = prepare_state_for_net(new_observation.copy()).to(device)
    
    policy, net_v = net(state_tensor)
    action_mask = torch.tensor(get_action_mask(node.state))
    p_vec = filter_legal_moves_and_renomalize(policy, action_mask)

    if recursion_count == 0:  # Add Dirichlet noise at root
        epsilon = 0.25
        alpha = 0.03
        num_legal = len(p_vec)
        noise = torch.distributions.Dirichlet(torch.full((num_legal,), alpha)).sample().to(device)
        p_vec = ((1 - epsilon) * p_vec) + (epsilon * noise)

    for move, p_val in zip(node.state.legal_moves, p_vec):
        next_state = deepcopy(node.state)
        next_state.push(move)
        next_state = next_state.mirror()
        new_node = Node(
            state=next_state,
            observation_tensor=new_observation,
            parent=node,
            prior=p_val.item()
        )
        node.children[move.uci()] = new_node

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
        sims: int = 1, 
        num_threads: int = 1,
        device: torch.device = None, 
        verbose: bool = False,
        inference_mode: bool = False
    ):
        
    if node is None: # if there is no subtree given to traverse, initialize root
        if state.turn:
            root = Node(state=state, observation_tensor=observation)
        if not state.turn: # if black's turn, we mirror the board to allow for easier translation of moes to actions
            root = Node(state=state.mirror(), observation_tensor=observation)
    else: # else provided node is the root of the tree
        root = node 

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
