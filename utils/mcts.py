import numpy as np
from copy import deepcopy
from utils.chess_utils_local import get_observation, legal_moves, result_to_int
from utils.utils import prepare_state_for_net, filter_legal_moves
import torch

# TODO make good comments
class Node:
    def __init__(self, state, p = 0):
        self.parent = None
        self.action_from_parent = None
        self.state = state
        self.children = []
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = 0

def backup(node, v=0):
    node.n += 1
    node.w += v
    node.q = node.w/node.n
    if node.parent is None: return
    # TODO what is virtual loss?
    return backup(node=node.parent, v=v)

@torch.no_grad()
def expand(node, net):
    # if game ends
    if node.state.is_insufficient_material() or node.state.can_claim_draw():
        v = result_to_int(result_string=node.state.outcome())

        return backup(node=node, v=v)

    else:
        c_puct = 1
        
        q_vec = np.array([child.q for child in node.children])
        n_vec = np.array([child.n for child in node.children])

        state_tensor = get_observation(orig_board=node.state, player=int(node.state.turn))
        # TODO actually handle board_history
        board_history = np.zeros((8, 8, 104), dtype=bool)
        state_tensor = np.dstack((state_tensor[:, :, :7], board_history))

        input(state_tensor.shape)

        state_tensor = prepare_state_for_net(state=state_tensor.copy())

        policy, v = net.forward(state_tensor)
        # 

        # TODO handle legal moves!
        legal_moves_iter = (legal_moves(orig_board=node.state))
        action_mask = np.zeros(4672, "int8")
        for i in legal_moves_iter:
            action_mask[i] = 1

        p_vec = filter_legal_moves(policy_vec=policy, legal_moves=action_mask)

        u_vec = c_puct*np.sqrt(sum(n_vec))*np.divide(p_vec, 1+n_vec)

        move = np.argmax(q_vec + u_vec)

        # TODO handle move output

        new_state = deepcopy(node.state)
        new_state.push(move)

        # if visited state already
        if new_state in [child.state for child in node.children]:
            return expand(node=child)

        else:
            #
            p = net(new_state)
            new_node = Node(state = new_state)
            node.children.append(new_node)
            return expand(node=new_node)
    
# TODO
def mcts(state, net, tau, sims=1):
    # state is a python-chess board    
    root = Node(
        state = state
    )

    for sim in range(sims):
        # TODO figure out how to parallelize
        expand(node=root, net=net)

    if tau == 0:
        # TODO argmax? 1/tau something something
        action = None
    elif tau > 0:
        N = np.array([child.n for child in root.children])**(1/tau)

        pi_vec =  N / sum(N)
        action = np.random.choice(a=[child.action_from_parent for child in root.children], p=pi_vec)

    # TODO resign criterion

    # TODO return 

    return action, value, 

