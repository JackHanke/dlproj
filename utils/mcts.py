import torch
from torch.distributions import Categorical
import chess
import threading
import logging
import numpy as np
import concurrent.futures
import gc
from copy import deepcopy

from utils.chess_utils_local import get_observation, legal_moves, result_to_int, get_action_mask, actions_to_moves, moves_to_actions, mirror_move
from utils.utils import prepare_state_for_net, filter_legal_moves_and_renomalize, renormalize_network_output, rand_argmax, observe

VIRTUAL_LOSS = 1
POSSIBLE_AGENTS = ['player_0', 'player_1']

# Node of search tree
class Node:
    def __init__(self, state: chess.Board, board_history: torch.tensor, agent: str, parent=None, prior: float = 0.0):
        self.state = state  # python-chess board object
        self.agent = agent
        self.board_history = board_history  # board history
        self.parent = parent  # parent node of Node, None if root Node
        self.children = {}  # dictionary of {chess.Move: child_Node}
        self.n = 0  # number of times Node has been visited
        self.w = 0  # total reward from this Node
        self.q = 0.0  # mean reward from this Node
        self.virtual_loss = 0  # virtual loss for node
        self.p = prior  # prior from network
        self.lock = threading.Lock()  # Per-node lock for thread safety

# Backup function for MCTS, loops until hitting the root node
def backup(node: Node, v: float = 0.0):
    while node is not None:
        with node.lock:  # Lock only when modifying node values
            node.n += 1
            node.w += v
            node.q = node.w / node.n
        v = -v  
        node = node.parent  # Move up the tree

@torch.no_grad()
def simulate(node: Node, net: torch.nn.Module, device: torch.device = None, recursion_count: int = 0, verbose: bool = False, c_puct: int = 3):
    # a) Selection - Traverse down the tree using UCB (NO LOCK HERE)
    if len(node.children) > 0:
        best_score = -float('inf')
        selected_move = None
        n = node.n + 1e-6  # Avoid division by zero

        for move in node.state.legal_moves:
            if move not in node.children:
                continue  # Should not happen

            child = node.children[move]
            q = child.q  # Safe to read without lock
            n_val = child.n  # Safe to read without lock
            p = child.p  # Safe to read without lock
            u = c_puct * np.sqrt(n) * p / (1 + n_val)  # UCB calculation
            score = q + u

            if score > best_score:
                best_score = score
                selected_move = move

        if selected_move is None:
            return backup(node, 0.0)

        next_node = node.children[selected_move]
        return simulate(next_node, net, device, recursion_count + 1, verbose, c_puct)

    # b) Expansion - Lock only when adding children
    if is_game_over(node.state):
        return backup(node, result_to_int(node.state.result(claim_draw=True)))

    new_observation = observe(board=node.state, agent=node.agent, possible_agents=POSSIBLE_AGENTS, board_history=node.board_history)['observation']
    state_tensor = prepare_state_for_net(new_observation.copy()).to(device)
    policy, net_v = net(state_tensor)
    action_mask = torch.tensor(get_action_mask(node.state))
    p_vec = filter_legal_moves_and_renomalize(policy, action_mask).squeeze(-1)

    with node.lock:  # Lock only when modifying node.children
        if len(node.children) == 0:
            for move, p_val in zip(node.state.legal_moves, p_vec):
                next_state = deepcopy(node.state)
                next_state.push(move)
                board_history = np.dstack((get_observation(next_state, player=0)[:, :, 7:], node.board_history[:, :, :-13]))
                
                new_node = Node(
                    state=next_state,
                    board_history=board_history,
                    agent=POSSIBLE_AGENTS[0] if node.agent == POSSIBLE_AGENTS[1] else POSSIBLE_AGENTS[1],
                    parent=node,
                    prior=p_val.item()
                )
                node.children[move] = new_node  # Add child safely

    return backup(node, net_v)

@torch.no_grad()
def mcts(
        state: chess.Board, 
        starting_agent: str,
        net: torch.nn.Module, 
        tau: int, 
        node: Node = None,
        c_puct: int = 3,
        sims: int = 1, 
        num_threads: int = 4,
        device: torch.device = None, 
        verbose: bool = False,
        inference_mode: bool = False
    ):
    net.to(device)
    board_history = np.zeros((8, 8, 104), dtype=bool)
    
    if node is None:
        root = Node(state=state, board_history=board_history, agent=starting_agent)
    else:
        root = node 

    # Expand tree with num_threads threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(simulate, root, net, device, 0, verbose, c_puct) for _ in range(sims)]
        _ = [f.result() for f in futures]

    # Construct pi from root node and given tau value
    pi = create_pi_vector(node=root, tau=tau)
    value = root.q
    m = Categorical(pi)

    if inference_mode: 
        sampled_action = int(pi.argmax(-1).item())
    else: 
        sampled_action = int(m.sample().item())

    root = None  # Dereference root for memory management
    gc.collect()

    return pi, value, sampled_action, None

def create_pi_vector(node: Node, tau: float):
    pi = [0 for _ in range(4672)]

    if tau == 0:
        highest_n, most_visited_action = 0, None
        for move, child in node.children.items():
            mirrored_move = mirror_move(move) if node.state.turn == chess.BLACK else move
            action = moves_to_actions[mirrored_move.uci()]
            if child.n > highest_n:
                highest_n = child.n
                most_visited_action = action
        pi[most_visited_action] = 1
        return torch.tensor(pi)

    pi_denom = sum([child.n**(1/tau) for child in node.children.values()])
    for move, child in node.children.items():
        mirrored_move = mirror_move(move) if node.state.turn == chess.BLACK else move
        action = moves_to_actions[mirrored_move.uci()]
        val = (child.n ** (1/tau)) / pi_denom
        pi[action] = val

    return torch.tensor(pi)

def is_game_over(board: chess.Board):
    next_legal_moves = legal_moves(board)
    return not any(next_legal_moves) or board.is_insufficient_material() or board.can_claim_draw()