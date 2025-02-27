import numpy as np
from copy import deepcopy
from utils.chess_utils_local import get_observation, legal_moves, result_to_int
from utils.utils import prepare_state_for_net, filter_legal_moves, renormalize_network_output
import torch
import time

# the node class holds a python-chess board object as a state
# the class points to its children (for the expansion phase of MCTS) and its parent (for the backup phase of MCTS)
# n is the number of times a sim has visited the node within a single MCTS
# w is the total action value, ie the total reward seen from this node
# q is the mean action value, ie the average reward seen from this node
class Node:
    def __init__(self, state, parent=None, action_from_parent=None):
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children = {}
        self.n = 0
        self.w = 0
        self.q = 0

# takes python-chess board and returns chosen move (in python-chess format)
def choose_move(node, net):
    # prepare board for network inference
    start = time.time()
    state_tensor = get_observation(orig_board=node.state, player=int(node.state.turn))
    # TODO actually handle board_history. Currently filling board history with zeros, but should instead track history of boards in sim
    board_history = np.zeros((8, 8, 104), dtype=bool)
    state_tensor = np.dstack((state_tensor[:, :, :7], board_history))
    state_tensor = prepare_state_for_net(state=state_tensor.copy())
    print(f'state_tensor formatting: {time.time()- start} s')

    # run network
    start = time.time()
    policy, v = net.forward(state_tensor)
    print(f'net evaluation: {time.time()- start} s')

    # compute legal move logits vector p_vec
    start = time.time()
    legal_moves_iter = (legal_moves(orig_board=node.state))
    action_mask = np.zeros(4672, "int8")
    for i in legal_moves_iter: action_mask[i] = 1
    action_mask = torch.tensor(action_mask)
    p_vec = renormalize_network_output(policy_vec=policy, legal_moves=action_mask)
    print(f'filter to legal moves: {time.time()- start} s')

    # make child nodes with the new move made
    start = time.time()
    for move in node.state.legal_moves:
        new_state = deepcopy(node.state)
        new_state.push(move)
        new_node = Node(state=new_state, parent=node, action_from_parent=move)
        node.children.append(new_node)
    print(f'make new nodes: {time.time()- start} s')

    # compute the best move using collected information
    # create vector of q and n values
    start = time.time()
    q_vec = torch.tensor([child.q for child in node.children]).unsqueeze(1)
    n_vec = torch.tensor([child.n for child in node.children]).unsqueeze(1)
    # compute u vector
    c_puct = 1
    u_vec = c_puct*torch.sqrt(sum(n_vec))*torch.divide(p_vec, 1+n_vec)
    # choose action
    chosen_move = np.argmax(q_vec + u_vec)
    print(f'choose action with statistics: {time.time()- start} s')

    return chosen_move

# compute is a board is in a terminal state (directly from PettingZoo source)
def is_game_over(board):
    # if game ends
    next_legal_moves = legal_moves(board)
    is_stale_or_checkmate = not any(next_legal_moves)

    # claim draw is set to be true to align with normal tournament rules
    is_insufficient_material = board.is_insufficient_material()
    can_claim_draw = board.can_claim_draw()
    game_over = can_claim_draw or is_stale_or_checkmate or is_insufficient_material
    return game_over

# backup function for MCTS, loops until hitting the root node (which is the node without a parent)
def backup(node, v=0):
    node.n += 1
    node.w += v
    node.q = node.w/node.n
    if node.parent is None: return
    # TODO what is virtual loss?
    return backup(node=node.parent, v=v)

@torch.no_grad()
def expand(node, net, recursion_count=0):
    # if the game has gone on too long, it's a draw
    if recursion_count > 60:
        # backup the sim with draw scoring
        return vanilla_backup(node=node, v=0, root_color=root_color)

    # if the node is a terminal state...
    elif is_game_over(board=node.state):
        # get the associated value of the state, ( 1, 0, -1 )
        v = result_to_int(result_str=node.state.result(claim_draw=True))
        # backup the sim
        return backup(node=node, v=v)

    # else if non-terminal state...
    else:
        # choose best move
        chosen_move = choose_move(node=node, net=net)
        # either the next node has already been seen...
        try:
            next_node = node.children[chosen_move]
        # or we initialize a new node...
        except KeyError:
            next_state = deepcopy(node.state)
            next_state.push(chosen_move)
            next_node = VanillaNode(
                state=next_state,
                parent=node,
                action_from_parent=chosen_move
            )
            node.children[chosen_move] = next_node
        # expand to chosen node
        return expand(node=next_node, net=net, recursion_count=recursion_count+1)
    
# TODO
def mcts(state, net, tau, sims=1):
    # state is a python-chess board    
    root = Node(state = state)

    # TODO figure out how to parallelize
    for sim in range(sims):
        start = time.time()
        expand(node=root, net=net)
        print(f'Total time for sim {sim}: {time.time()-start} s')

    if tau == 0:
        # TODO argmax? 1/tau something something
        action = None
    elif tau > 0:
        N = np.array([child.n for child in root.children])**(1/tau)

        # TODO this is not correct!!! need to add 'u' somehow
        pi = N / sum(N)
        # pi_vec = torch.tensor([child.q for child in root.children])
        action = np.random.choice(a=[child.action_from_parent for child in root.children], p=pi)

    value = root.q

    # TODO resign criterion

    # TODO return 
    return pi, value

