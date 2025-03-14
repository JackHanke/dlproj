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
def expand(
        node: Node, 
        net: torch.nn.Module, 
        device: torch.device = None, 
        recursion_count: int = 0, 
        verbose: bool = False,
        c_puct: int = 3,
    ):
    # if the node is a game over state
    if is_game_over(board=node.state):
        # get the associated value of the state, ( 1, 0, -1 )
        v = result_to_int(result_str=node.state.result(claim_draw=True))
        # backup the sim
        return backup(node=node, v=v)

    # if the game has gone on too long, it's a draw TODO delete this check?
    elif recursion_count > 60:
        # backup the sim with draw scoring
        return backup(node=node, v=0.0)

    # if this is a new node, create all children with priors
    if len(node.children.keys()) == 0:
        # TODO do I need a lock to get the state?
        next_board = get_observation(orig_board=node.state, player=0) # NOTE no flipping the observation

        # create observation tensor with proper board history
        new_observation = np.dstack((next_board[:, :, 7:], node.observation_tensor[:, :, :-13]))
        state_tensor = prepare_state_for_net(state=new_observation.copy()).to(device)
        
        policy, net_v = net.forward(state_tensor)
        # backup v
        action_mask = torch.tensor(get_action_mask(orig_board=node.state))
        p_vec = filter_legal_moves_and_renomalize(policy_vec=policy, legal_moves=action_mask)

        if recursion_count == 0:
            # Inject Dirichlet noise at the root
            epsilon = 0.25
            alpha = 0.03
            num_legal = len(p_vec)
            noise = torch.distributions.Dirichlet(torch.full((num_legal,), alpha)).sample().unsqueeze(1).to(device)
            p_vec = ((1 - epsilon) * p_vec) + (epsilon * noise)

        for i, move in enumerate(node.state.legal_moves):
            # if move.uci() not in node.children:
            next_state = deepcopy(node.state)
            next_state.push(move)
            next_state = next_state.mirror()
            # Set the prior for this move to the corresponding network output.
            new_node = Node(
                state=next_state,
                observation_tensor=new_observation,
                parent=node, 
                prior=p_vec[i].item()
            )
            with threading.Lock():
                node.children[move.uci()] = new_node
        # because we were doing things wrong
        return backup(node=node, v=net_v)

    # with priors (either already there or previously calculated!)
    best_score = -float('inf')
    selected_move = None
    n = node.n + 1e-6
    for move in node.state.legal_moves:
        with threading.Lock():
            try:
                child = node.children[move.uci()]
            except KeyError as e:
                logging.error(e)
                logging.error(f'Board player: {node.state.turn}')
                logging.error(f'Legal moves : {[thing.uci() for thing in node.state.legal_moves]}')
                logging.error(f'  Children  : {[thing for thing in node.children]}')
                input('Illegal move entered. Possible race condition error.')
        # with child.lock:
            q = child.q
            n_val = child.n
            p = child.p
        u = c_puct * np.sqrt(n) * p / (1 + n_val)
        score = q + u
        if score > best_score:
            best_score = score
            selected_move = move
    
    with threading.Lock():
        next_node = node.children[selected_move.uci()]

    # TODO virtual loss add stuff?

    return expand(
        node=next_node, 
        net=net, 
        device=device,
        recursion_count=recursion_count+1, 
        verbose=verbose
    )

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
        futures = [executor.submit(expand, root, net, device, 0, verbose, c_puct) for _ in range(sims)]
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
