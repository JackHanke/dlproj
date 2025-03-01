import numpy as np
from copy import deepcopy
from utils.chess_utils_local import get_observation, legal_moves, result_to_int, get_action_mask
from utils.utils import prepare_state_for_net, filter_legal_moves, filter_legal_moves_and_renomalize, renormalize_network_output, rand_argmax
import torch
import time
import chess

# NOTE this code refers to a chess.Move and a UCI string as a 'move'
# while the action indexes provided by Pettingzoo as an 'action'

# node of Monte Carlo Tree
class Node:
    def __init__(
            self, 
            state: chess.Board, 
            parent = None, 
            action_from_parent: chess.Move = None
        ):
        self.state = state # python-chess board object
        self.parent = parent # parent node of Node, None if root Node
        self.action_from_parent = action_from_parent # chess.Move for action taken to arrive at Node, None for root
        self.children = {} # dictionary of {chess.Move: child_Node}
        self.n = 0 # number of times Node has been visited
        self.w = 0 # total reward from this Node
        self.q = 0 # mean reward from this Node

# creates the final pi tensor (same size as network output)
def create_pi_vector(node: type[Node], tau: float):
    pi = [0 for _ in range(4672)]
    action_mask = torch.tensor(get_action_mask(orig_board=node.state))
    legal_action_indexes = torch.nonzero(action_mask)

    legal_moves_uci = [move_obj.uci() for move_obj in node.state.legal_moves]

    if tau == 0:
        most_visited_action, highest_visit_count = 0, 0
        for move, action in zip(legal_moves_uci, legal_action_indexes):
            try:
                action = action.item()
                visit = (node.children[chess.Move.from_uci(move)].n)
                if visit > highest_visit_count:
                    highest_visit_count = visit
                    most_visited_action = action
            except KeyError:
                pass
        pi[action] = 1
        pi = torch.tensor(pi).unsqueeze(0)    
        return pi

    pi_denom = sum([child.n**(1/tau) for child in node.children.values()])

    for move, action in zip(legal_moves_uci, legal_action_indexes):
        action = action.item()
        try:
            val = (node.children[chess.Move.from_uci(move)].n ** (1/tau))/pi_denom
            pi[action] = val
        except KeyError:
            pi[action] = 0
    pi = torch.tensor(pi).unsqueeze(0)
    return pi

# takes python-chess board and returns chosen move (in python-chess format)
@torch.no_grad()
def choose_move(node: type[Node], net: torch.nn.Module, recursion_count: int, verbose: bool = False):
    # prepare board for network inference
    start = time.time()
    # NOTE it is possible player=node.state.turn or something like that, im not sure
    state_tensor = get_observation(orig_board=node.state, player=0)
    action_mask = torch.tensor(get_action_mask(orig_board=node.state))

    legal_action_indexes = torch.nonzero(action_mask)
    legal_moves_uci = [move_obj.uci() for move_obj in node.state.legal_moves]

    # TODO actually handle board_history. Currently filling board history with zeros, but should instead track history of boards in sim
    board_history = np.zeros((8, 8, 104), dtype=bool)
    state_tensor = np.dstack((state_tensor[:, :, :7], board_history))
    state_tensor = prepare_state_for_net(state=state_tensor.copy())
    if verbose: print(f'state_tensor formatting: {time.time()- start} s')

    # TODO send state to network for inference
    # run network
    start = time.time()
    policy, v = net.forward(state_tensor)
    if verbose: print(f'net evaluation: {time.time()- start} s')

    # compute legal move logits vector p_vec
    start = time.time()
    # network-friendly numbers
    p_vec = filter_legal_moves_and_renomalize(policy_vec=policy, legal_moves=action_mask)
    # if the root node, disturb the choice with noise for variety of search
    if recursion_count == 0:
        alpha = 0.03
        dist = torch.distributions.dirichlet.Dirichlet(torch.tensor([alpha for _ in range(p_vec.shape[0])]))
        eta = dist.sample().unsqueeze(1)
        epsilon = 0.25
        p_vec = ((1-epsilon) * p_vec) + (epsilon * eta)

    if verbose: print(f'filter to legal moves: {time.time()- start} s')

    # access nodes or spoof values 
    start = time.time()
    q_vec, n_vec = [], []
    for move in legal_moves_uci:
        # get values if they exist
        try:
            child_node = node.children[move]
            q_vec.append(child_node.q)
            n_vec.append(child_node.n)
        # otherwise they are zero
        except KeyError:
            q_vec.append(0)
            n_vec.append(0)

    q_vec = torch.tensor(q_vec).unsqueeze(1)
    n_vec = torch.tensor(n_vec).unsqueeze(1)
    if verbose: print(f'make q and n vectors: {time.time()- start} s')

    # compute u vector
    c_puct = 1
    u_vec = c_puct*torch.sqrt(sum(n_vec))*torch.divide(p_vec, 1+n_vec)
    # choose move
    chosen_move = chess.Move.from_uci(legal_moves_uci[rand_argmax(q_vec + u_vec)])

    return chosen_move

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
def backup(node: type[Node], v: int = 0):
    node.n += 1
    node.w += v
    node.q = node.w/node.n
    # 
    if node.parent is None: return
    # TODO what is virtual loss?
    return backup(node=node.parent, v=v)

@torch.no_grad()
def expand(node: type[Node], net: torch.nn.Module, recursion_count: int = 0, verbose: bool = False):
    # if the game has gone on too long, it's a draw TODO delete this check?
    if recursion_count > 60:
        # backup the sim with draw scoring
        return backup(node=node, v=0)

    # if the node is a terminal state...
    elif is_game_over(board=node.state):
        # get the associated value of the state, ( 1, 0, -1 )
        v = result_to_int(result_str=node.state.result(claim_draw=True))
        # backup the sim
        return backup(node=node, v=v)

    # else if non-terminal state...
    else:
        # choose best move, in the python-chess format
        chosen_move = choose_move(node=node, net=net, recursion_count=recursion_count, verbose=verbose)
        # either the next node has already been seen...
        try:
            next_node = node.children[chosen_move]
        # or we initialize a new node...
        except KeyError:
            next_state = deepcopy(node.state)
            next_state.push(chosen_move)
            next_node = Node(
                state = next_state,
                parent = node,
                action_from_parent = chosen_move
            )
            node.children[chosen_move] = next_node
        # expand to chosen node
        return expand(node=next_node, net=net, recursion_count=recursion_count+1, verbose=verbose)
    
# conduct Monte Carlo Tree Search for sims sims and the python-chess state
# using network net and temperature tau
def mcts(state: chess.Board, net: torch.nn.Module, tau: int, sims: int = 1, verbose: bool = False):
    # state is a python-chess board    
    root = Node(state=state)

    # TODO figure out how to parallelize
    tot_start = time.time()
    for sim in range(sims):
        start = time.time()
        expand(node=root, net=net, verbose=verbose)
        if verbose: print(f'Total time for sim {sim}: {time.time()-start} s')
    if verbose: print(f'Total time for {sims} sims: {time.time()-tot_start} s')

    # Turn data from tree into pi vector
    pi = create_pi_vector(node=root, tau=tau)
    # value is the mean value from the root
    value = root.q
    # get best value calculated from pi
    chosen_action = int(torch.argmax(pi))
    return pi, value, chosen_action

