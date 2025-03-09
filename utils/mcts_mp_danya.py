import os
import logging
import numpy as np
from copy import deepcopy
import torch
import torch.multiprocessing as mp
import time
import chess
from utils.chess_utils_local import get_observation, legal_moves, result_to_int, get_action_mask


# Set up logging configuration.
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [PID %(process)d] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

# Global variable for the shared manager; set in worker initializer.
global_manager = None

def init_worker(mgr):
    global global_manager
    global_manager = mgr
    logging.debug("Worker initialized with shared manager.")

# --- Shared Memory Node Class ---
class Node:
    def __init__(self, state: chess.Board, manager, parent=None, action_from_parent: chess.Move = None):
        self.state = state                              # chess.Board object
        self.parent = parent                            # parent node (None if root)
        self.action_from_parent = action_from_parent    # move that led to this node
        # Create shared objects using the manager:
        self.children = manager.dict()                  # dictionary: move (uci string) -> child Node
        self.n = manager.Value('i', 0)                  # visit count
        self.w = manager.Value('d', 0.0)                # total reward
        self.q = manager.Value('d', 0.0)                # mean reward
        self.virtual_loss = manager.Value('i', 0)       # virtual loss
        self.lock = manager.Lock()                      # lock for concurrent updates

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'lock' in state:
            del state['lock']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        global global_manager
        # Reinitialize the lock using the global_manager.
        if global_manager is not None:
            self.lock = global_manager.Lock()
        else:
            self.lock = mp.Lock()

# --- Helper Functions ---
def create_pi_vector(node: Node, tau: float):
    pi = [0 for _ in range(4672)]
    action_mask = torch.tensor(get_action_mask(orig_board=node.state))
    legal_action_indexes = torch.nonzero(action_mask)

    legal_moves_uci = [move_obj.uci() for move_obj in node.state.legal_moves]

    if tau == 0:
        most_visited_action, highest_visit_count = 0, 0
        for move, action in zip(legal_moves_uci, legal_action_indexes):
            try:
                action = action.item()
                # Access the integer value using .value
                visit = node.children[chess.Move.from_uci(move)].n.value
                if visit > highest_visit_count:
                    highest_visit_count = visit
                    most_visited_action = action
            except KeyError:
                pass
        pi[most_visited_action] = 1
        pi = torch.tensor(pi).unsqueeze(0)    
        return pi

    # Use .value for the exponentiation operation
    pi_denom = sum([child.n.value**(1/tau) for child in node.children.values()])

    for move, action in zip(legal_moves_uci, legal_action_indexes):
        action = action.item()
        try:
            val = (node.children[chess.Move.from_uci(move)].n.value ** (1/tau)) / pi_denom
            pi[action] = val
        except KeyError:
            pi[action] = 0
    pi = torch.tensor(pi).unsqueeze(0)
    return pi


@torch.no_grad()
def choose_move(node: Node, net: torch.nn.Module, recursion_count: int, verbose: bool = False):
    legal_moves = list(node.state.legal_moves)
    if not legal_moves:
        return None
    move = legal_moves[0]
    if verbose:
        logging.debug(f"Recursion {recursion_count}: Process {os.getpid()} chosen move {move.uci()}")
    return move

def is_game_over(board: chess.Board):
    return board.is_game_over()

def expand(node: Node, net: torch.nn.Module, manager, recursion_count: int = 0, verbose: bool = False, virtual_loss: int = 1):
    pid = os.getpid()
    logging.debug(f"Process {pid} starting expand at recursion {recursion_count}.")
    if recursion_count > 60:
        logging.debug(f"Process {pid} reached maximum recursion at level {recursion_count}.")
        return 0
    if is_game_over(node.state):
        v = result_to_int(result_str=node.state.result(claim_draw=True))
        logging.debug(f"Process {pid} reached terminal node at recursion {recursion_count}.")
        return 1  # Dummy terminal value

    with node.lock:
        node.n.value += 1
        node.virtual_loss.value += virtual_loss
        logging.debug(f"Process {pid} updated node at recursion {recursion_count}: n={node.n.value}, virtual_loss={node.virtual_loss.value}.")

    move = choose_move(node, net, recursion_count, verbose)
    if move is None:
        logging.debug(f"Process {pid} no legal move at recursion {recursion_count}.")
        return 0
    move_key = move.uci()

    with node.lock:
        if move_key in node.children:
            next_node = node.children[move_key]
            logging.debug(f"Process {pid} found existing child for move {move_key} at recursion {recursion_count}.")
        else:
            next_state = deepcopy(node.state)
            next_state.push(move)
            next_node = Node(next_state, manager, parent=node, action_from_parent=move)
            node.children[move_key] = next_node
            logging.debug(f"Process {pid} created new child for move {move_key} at recursion {recursion_count}.")

    v = expand(next_node, net, manager, recursion_count + 1, verbose, virtual_loss)

    with node.lock:
        node.virtual_loss.value -= virtual_loss
        node.w.value += v
        node.q.value = node.w.value / node.n.value
        logging.debug(f"Process {pid} backed up node at recursion {recursion_count}: w={node.w.value}, q={node.q.value}.")

    return v

def mcts_worker(root: Node, net: torch.nn.Module, manager, sims: int, verbose: bool):
    pid = os.getpid()
    logging.debug(f"Worker process {pid} starting with {sims} simulations.")
    for _ in range(sims):
        expand(root, net, manager, recursion_count=0, verbose=verbose)
    logging.debug(f"Worker process {pid} finished simulations.")

def mcts(state: chess.Board, net: torch.nn.Module, tau: int, total_sims: int = 100, num_workers: int = 4, verbose: bool = False):
    mp.set_start_method('fork', force=True)
    manager = mp.Manager()
    # Set up the shared root node.
    root = Node(state, manager)
    sims_per_worker = total_sims // num_workers

    processes = []
    for _ in range(num_workers):
        p = mp.Process(target=mcts_worker, args=(root, net, manager, sims_per_worker, verbose))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    pi = create_pi_vector(root, tau)
    value = root.q.value
    chosen_action = int(torch.argmax(pi))
    logging.debug(f"MCTS complete. Final node: n={root.n.value}, w={root.w.value}, q={root.q.value}.")
    return pi, value, chosen_action