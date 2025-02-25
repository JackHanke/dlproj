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

def backup(node):
    node.n += 1
    node.w += v
    node.q = node.w/node.n
    if node.parent is None: return
    # TODO virtual loss?
    return backup(node=node.parent)

def expand(node):
    # if game ends
    if node.state.is_insufficient_material() or node.state.can_claim_draw():
        return backup(node=node)

    else:
        c_puct = 1
        
        q_vec = numpy.array([child.q for child in node.children])
        n_vec = numpy.array([child.n for child in node.children])

        probs = net.forward(node.state)
        p_vec = filter_legal_moves(probs)

        u_vec = c_puct*np.sqrt(sum(n_vec))*np.divide(p_vec, 1+n_vec)

        move = np.argmax(q_vec + u_vec)

        new_state = node.state.copy()
        new_state.push(move)

        # if visited state already
        if new_state in [child.state in ]:
            expand(node=child)

        else:
            #         
            new_node = Node(
                state = new_state,
                p = 0
            )
            node.children.append(new_node)
            expand(node=new_node)
    

def mcts(state, net, tau, sims=1):
    # state is a python-chess board    
    root = Node(
        state = state
    )

    for sim in range(sims):
        # TODO figure out how to parallelize
        expand(node=root)

    if tau == 0:
        # TODO argmax? 1/tau something something
        action = None
    elif tau > 0:
        N = np.array([child.n for child in root.children])**(1/tau)

        pi_vec =  N / sum(N)
        action = np.random.choice(a=[child.action_from_parent for child in root.children], p=pi_vec)

    # TODO resign criterion

    return action, value
