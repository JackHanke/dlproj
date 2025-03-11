import numpy as np
import torch
import time
import chess
import threading
import concurrent.futures
from copy import deepcopy
from utils.chess_utils_local import get_observation, legal_moves, result_to_int, get_action_mask
from utils.utils import prepare_state_for_net, filter_legal_moves_and_renomalize, rand_argmax

# Constant for virtual loss (if using parallelization)
VIRTUAL_LOSS = 1

# Updated Node class now stores the prior probability p
class Node:
    def __init__(self, state: chess.Board, parent=None, action_from_parent: chess.Move = None, prior=0.0):
        self.state = state                      # chess.Board object
        self.parent = parent                    # Parent Node (None if root)
        self.action_from_parent = action_from_parent  # Move that led here
        self.children = {}                      # Mapping from chess.Move to Node
        self.n = 0                              # Visit count
        self.w = 0                              # Total reward
        self.q = 0                              # Mean reward (w/n)
        self.p = prior                          # Prior probability from network
        self.lock = threading.Lock()            # For thread safety

# Create the final pi vector (of same length as network output)
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
                visit = node.children[chess.Move.from_uci(move)].n
                if visit > highest_visit_count:
                    highest_visit_count = visit
                    most_visited_action = action
            except KeyError:
                pass
        pi[most_visited_action] = 1
        pi = torch.tensor(pi).unsqueeze(0)
        return pi

    pi_denom = sum([child.n**(1/tau) for child in node.children.values()])
    for move, action in zip(legal_moves_uci, legal_action_indexes):
        action = action.item()
        try:
            val = (node.children[chess.Move.from_uci(move)].n ** (1/tau)) / pi_denom
            pi[action] = val
        except KeyError:
            pi[action] = 0
    pi = torch.tensor(pi).unsqueeze(0)
    return pi

# Check if the game is over
def is_game_over(board: chess.Board):
    next_legal_moves = legal_moves(board)
    is_stale_or_checkmate = not any(next_legal_moves)
    is_insufficient_material = board.is_insufficient_material()
    can_claim_draw = board.can_claim_draw()
    return can_claim_draw or is_stale_or_checkmate or is_insufficient_material

# The main simulation function
def simulate(root: Node, net: torch.nn.Module, verbose: bool = False):
    """
    One simulation: traverse the tree, expand a leaf fully, evaluate it,
    and backpropagate the value while handling virtual loss.
    """
    path = []  # To record the traversal path as (node, move) pairs.
    node = root

    # --------------------------
    # Selection Phase
    # --------------------------
    while True:
        if is_game_over(node.state):
            break

        legal_moves_list = list(node.state.legal_moves)
        # If the node is not fully expanded, stop here.
        if len(legal_moves_list) > len(node.children):
            # Stop at the first unexpanded move.
            break
        # otherwise we have expanded at this node
        else:
            # Node is fully expanded: select the child that maximizes Q + U.
            # total_n = sum(child.n for child in node.children.values())
            total_n = node.n
            best_score = -float('inf')
            selected_move = None
            c_puct = 1  # exploration constant
            for move in legal_moves_list:
                child = node.children[move]
                with child.lock:
                    q = child.q
                    n_val = child.n
                    p = child.p     
                u = c_puct * np.sqrt(total_n) * p / (1 + n_val)
                score = q + u
                if score > best_score:
                    best_score = score
                    selected_move = move

            path.append((node, selected_move)) # TODO fix this
            # Apply virtual loss for parallelism.
            # with node.lock:
            #     if selected_move not in node.children:
            #         print('this happens at least once !!!!!!!!!!!!!')
            #         next_state = deepcopy(node.state)
            #         next_state.push(selected_move)
            #         new_node = Node(state=next_state, parent=node, action_from_parent=selected_move, prior=0)
            #         node.children[selected_move] = new_node
            child = node.children[selected_move]
            with child.lock:
                child.n += VIRTUAL_LOSS
                child.w -= VIRTUAL_LOSS # NOTE wtf
                child.q = child.w / child.n if child.n != 0 else 0 # NOTE hehehehehehe
            node = child

    # --------------------------
    # Evaluation / Full Expansion Phase
    # --------------------------
    if is_game_over(node.state):
        v = result_to_int(node.state.result(claim_draw=True))
    else:
        # Prepare state tensor for network evaluation.

        # TODO not even kidding, we need to track board history in the state tensor
        state_tensor = get_observation(orig_board=node.state, player=0)
        board_history = np.zeros((8, 8, 104), dtype=bool)
        state_tensor = np.dstack((state_tensor[:, :, :7], board_history))
        state_tensor = prepare_state_for_net(state=state_tensor.copy())
        
        with torch.no_grad():
            policy, _ = net.forward(state_tensor)
        action_mask = torch.tensor(get_action_mask(orig_board=node.state))

        # Compute a vector of legal-move probabilities.
        p_vec = filter_legal_moves_and_renomalize(policy_vec=policy, legal_moves=action_mask)

        # NOTE this assumes that the order of node.state.legal_moves is the same as the outputted p_vec TODO CHECK
        legal_moves_list = list(node.state.legal_moves)
        for i, move in enumerate(legal_moves_list):
            with node.lock:
                if move not in node.children:
                    next_state = deepcopy(node.state)
                    next_state.push(move)
                    # Set the prior for this move to the corresponding network output.
                    new_node = Node(state=next_state, parent=node, action_from_parent=move, prior=p_vec[i].item())
                    node.children[move] = new_node

    # --------------------------
    # Backup Phase (remove virtual loss and update statistics)
    # --------------------------
    for parent, move in reversed(path):
        with parent.lock:
            child = parent.children[move]
            with child.lock:
                # Remove virtual loss adjustments.
                child.n -= VIRTUAL_LOSS
                child.w += VIRTUAL_LOSS
            parent.n += 1
            parent.w += v
            parent.q = parent.w / parent.n if parent.n != 0 else 0
        v = -v  # Flip the value for the alternate perspective.

    return

# Parallel MCTS: run multiple simulations concurrently.
@torch.no_grad()
def parallel_mcts(state: chess.Board, net: torch.nn.Module, tau: int, sims: int = 100, verbose: bool = False, num_threads: int = 4, device: torch.device = None):
    net.eval()
    root = Node(state=state)
    # Expand root to set its priors and inject Dirichlet noise exactly once.
    state_tensor = get_observation(orig_board=root.state, player=0)
    board_history = np.zeros((8, 8, 104), dtype=bool)
    state_tensor = np.dstack((state_tensor[:, :, :7], board_history))
    state_tensor = prepare_state_for_net(state=state_tensor.copy())

    # net.to(device)
    policy, _ = net.forward(state_tensor)
    action_mask = torch.tensor(get_action_mask(orig_board=root.state))
    p_vec = filter_legal_moves_and_renomalize(policy_vec=policy, legal_moves=action_mask)
    # Inject Dirichlet noise at the root.
    epsilon = 0.25
    alpha = 0.03
    num_legal = len(p_vec)
    noise = torch.distributions.Dirichlet(torch.full((num_legal,), alpha)).sample().unsqueeze(1)
    p_vec = ((1 - epsilon) * p_vec) + (epsilon * noise)
    legal_moves_list = list(root.state.legal_moves)
    for i, move in enumerate(legal_moves_list):
        next_state = deepcopy(root.state)
        next_state.push(move)
        # Initialize child nodes with the noisy prior.
        root.children[move] = Node(state=next_state, parent=root, action_from_parent=move, prior=p_vec[i].item())
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(simulate, root, net, verbose) for _ in range(sims)]
        _ = [f.result() for f in futures]
    
    pi = create_pi_vector(node=root, tau=tau)
    value = root.q
    print("Root stats: q =", root.q, ", n =", root.n, ", w =", root.w)
    chosen_action = int(torch.argmax(pi))
    return pi, value.item(), chosen_action