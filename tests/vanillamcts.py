import numpy as np
import torch
import time
from copy import deepcopy
from utils.chess_utils_local import get_observation, legal_moves, result_to_int
from utils.utils import prepare_state_for_net, filter_legal_moves, renormalize_network_output

# takes python-chess board and returns chosen move (in python-chess format)
def choose_move(node):
    board = node.state
    legal_moves = np.array(list(board.legal_moves))
    chosen_move = np.random.choice(legal_moves)
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

## Trying Vanilla MCTS first
class VanillaNode:
    def __init__(self, state, parent=None, action_from_parent=None):
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children = {}
        self.n = 0
        self.wins_n = 0

def vanilla_backup(node, root_color, v=0):
    node.n += 1
    if v*root_color == 1: node.wins_n += 1
    elif v == 0: node.wins_n += 0.5
    if node.parent is None: return
    return vanilla_backup(node=node.parent, v=v, root_color=root_color)

@torch.no_grad()
def vanilla_expand(node, root_color, recursion_count=0):
    # if the game has gone on too long, it's a draw
    if recursion_count > 60:
        # backup the sim with draw scoring
        return vanilla_backup(node=node, v=0, root_color=root_color)

    # if the node is a terminal state...
    elif is_game_over(board=node.state):
        # get the associated value of the state, ( 1, 0, -1 )
        v = result_to_int(result_str=node.state.result(claim_draw=True))
        # backup the sim
        return vanilla_backup(node=node, v=v, root_color=root_color)

    # else if non-terminal state...
    else:
        # choose best move
        chosen_move = choose_move(node=node)
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
        return vanilla_expand(node=next_node, root_color=root_color, recursion_count=recursion_count+1)

def vanilla_mcts(state, sims):
    root = VanillaNode(state = state)

    # TODO figure out how to parallelize
    for sim in range(sims):
        start = time.time()
        vanilla_expand(node=root, root_color=2*int(state.turn)-1)
        # print(f'Total time for sim {sim}: {time.time()-start} s')

    # print(f'Denominator at root: {root.n}')
    for action, next_node in root.children.items():
        print(f'taking action pi_{action}= {next_node.wins_n/next_node.n}')
    
    # choose final action
    pi = []
    for move in state.legal_moves:
        try:
            next_node = root.children[move]
            pi.append(next_node.wins_n/next_node.n)
        except KeyError:
            pass

    value = 1
    # action = 
    return pi, value