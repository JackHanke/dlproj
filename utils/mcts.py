import utils.chess_utils_local
import numpy as np

class Node:
    def __init__(self, state, p = 0):
        self.parent = None
        self.action_from_parent = None
        self.state = state
        self.children = None
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = 0

def backup(node, v=0):
    node.n += 1
    node.w += v
    node.q = node.w/node.n
    if node.parent is None: return
    # TODO virtual loss?
    return backup(node=node.parent, v=v)

def expand(node, net):
    # if game ends
    if node.state.is_insufficient_material() or node.state.can_claim_draw():
        result_string = node.state.outcome()
        if result_string == '1-0': v = 1
        elif result_string == '0-1': v = -1
        elif result_string == '1/2-1/2': v = 0

        return backup(node=node, v=v)

    else:
        c_puct = 1
        
        q_vec = np.array([child.q for child in node.children])
        n_vec = np.array([child.n for child in node.children])

        probs = net.forward(node.state)
        # 
        p_vec = filter_legal_moves(probs)

        u_vec = c_puct*np.sqrt(sum(n_vec))*np.divide(p_vec, 1+n_vec)

        move = np.argmax(q_vec + u_vec)

        # TODO handle move output

        new_state = node.state.copy()
        new_state.push(move)

        # if visited state already
        if new_state in [child.state for child in node.children]:
            expand(node=child)

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



