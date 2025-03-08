import numpy as np
from copy import deepcopy
import torch
import torch.multiprocessing as mp
import time
import chess
from utils.chess_utils_local import get_observation, legal_moves, result_to_int, get_action_mask
from utils.utils import prepare_state_for_net, filter_legal_moves_and_renomalize, rand_argmax

# --- Node Class with Virtual Loss and Lock ---
class Node:
    def __init__(self, state: chess.Board, parent=None, action_from_parent: chess.Move = None):
        self.state = state                              # python-chess board object
        self.parent = parent                            # parent node (None if root)
        self.action_from_parent = action_from_parent    # move taken to reach this node
        self.children = {}                              # dictionary: move -> child Node
        self.n = 0                                      # visit count
        self.w = 0                                      # total reward (wins)
        self.q = 0                                      # mean reward (w/n)
        self.virtual_loss = 0                           # temporary loss during simulation
        self.lock = mp.Lock()                           # lock for safe concurrent updates

# --- Helper Functions ---

def create_pi_vector(node: Node, tau: float):
    """Create a probability vector (pi) from the visit counts."""
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
        return torch.tensor(pi).unsqueeze(0)
    
    pi_denom = sum([child.n**(1/tau) for child in node.children.values()])
    for move, action in zip(legal_moves_uci, legal_action_indexes):
        action = action.item()
        try:
            val = (node.children[chess.Move.from_uci(move)].n ** (1/tau)) / pi_denom
            pi[action] = val
        except KeyError:
            pi[action] = 0
    return torch.tensor(pi).unsqueeze(0)

@torch.no_grad()
def choose_move(node: Node, net: torch.nn.Module, recursion_count: int, verbose: bool = False):
    """
    Choose a move based on network evaluation and current node statistics.
    The Q value used for selection is adjusted by subtracting the virtual loss.
    """
    start = time.time()
    state_tensor = get_observation(orig_board=node.state, player=0)
    action_mask = torch.tensor(get_action_mask(orig_board=node.state))
    legal_action_indexes = torch.nonzero(action_mask)
    legal_moves_uci = [move_obj.uci() for move_obj in node.state.legal_moves]

    # TODO: Improve board history handling.
    board_history = np.zeros((8, 8, 104), dtype=bool)
    state_tensor = np.dstack((state_tensor[:, :, :7], board_history))
    state_tensor = prepare_state_for_net(state=state_tensor.copy())
    if verbose:
        print(f'state_tensor formatting: {time.time() - start} s')

    start = time.time()
    net.eval()
    policy, v = net.forward(state_tensor)
    if verbose:
        print(f'net evaluation: {time.time() - start} s')

    start = time.time()
    p_vec = filter_legal_moves_and_renomalize(policy_vec=policy, legal_moves=action_mask)
    if recursion_count == 0:
        alpha = 0.03
        dist = torch.distributions.dirichlet.Dirichlet(torch.tensor([alpha] * p_vec.shape[0]))
        eta = dist.sample().unsqueeze(1)
        epsilon = 0.25
        p_vec = ((1 - epsilon) * p_vec) + (epsilon * eta)
    if verbose:
        print(f'filter to legal moves: {time.time() - start} s')

    q_vec, n_vec = [], []
    for move in legal_moves_uci:
        try:
            child_node = node.children[move]
            # Adjust Q by subtracting the virtual loss.
            adjusted_q = ((child_node.w - child_node.virtual_loss) / child_node.n) if child_node.n > 0 else 0
            q_vec.append(adjusted_q)
            n_vec.append(child_node.n)
        except KeyError:
            q_vec.append(0)
            n_vec.append(0)
    q_vec = torch.tensor(q_vec).unsqueeze(1)
    n_vec = torch.tensor(n_vec).unsqueeze(1)

    c_puct = 1
    u_vec = c_puct * torch.sqrt(sum(n_vec)) * torch.divide(p_vec, 1 + n_vec)
    chosen_move = chess.Move.from_uci(legal_moves_uci[rand_argmax(q_vec + u_vec)])
    return chosen_move

def is_game_over(board: chess.Board):
    """Determine if the board state is terminal."""
    next_legal_moves = legal_moves(board)
    is_stale_or_checkmate = not any(next_legal_moves)
    is_insufficient_material = board.is_insufficient_material()
    can_claim_draw = board.can_claim_draw()
    return can_claim_draw or is_stale_or_checkmate or is_insufficient_material

@torch.no_grad()
def expand(node: Node, net: torch.nn.Module, recursion_count: int = 0, verbose: bool = False, virtual_loss: int = 1):
    """
    Recursively expand the tree.
    Before expanding a node, a virtual loss is applied (and later removed) to penalize
    nodes currently being explored by other threads. The value is then backed up inline.
    """
    # Terminal conditions: if depth exceeds a threshold or game is over.
    if recursion_count > 60:
        return 0
    if is_game_over(node.state):
        v = result_to_int(result_str=node.state.result(claim_draw=True))
        return v

    # Reserve the node by applying virtual loss.
    with node.lock:
        node.n += 1
        node.virtual_loss += virtual_loss

    chosen_move = choose_move(node, net, recursion_count, verbose)

    try:
        next_node = node.children[chosen_move]
    except KeyError:
        next_state = deepcopy(node.state)
        next_state.push(chosen_move)
        next_node = Node(state=next_state, parent=node, action_from_parent=chosen_move)
        with node.lock:
            node.children[chosen_move] = next_node

    # Recursively expand the chosen child node.
    v = expand(next_node, net, recursion_count + 1, verbose, virtual_loss)

    # Remove the virtual loss and update the node's statistics.
    with node.lock:
        node.virtual_loss -= virtual_loss
        node.w += v
        node.q = node.w / node.n

    return v

def mcts_worker(root: Node, net: torch.nn.Module, sims: int, verbose: bool = False):
    """Worker function for running MCTS simulations in parallel."""
    for _ in range(sims):
        expand(root, net, verbose=verbose)

def mcts(state: chess.Board, net: torch.nn.Module, tau: int, total_sims: int = 100, num_workers: int = 4, verbose: bool = False):
    """
    Conduct MCTS with parallel simulations using torch.multiprocessing.
    Multiple workers concurrently expand the shared root.
    """
    root = Node(state=state)
    sims_per_worker = total_sims // num_workers

    mp.set_start_method('spawn', force=True)
    processes = []
    for _ in range(num_workers):
        p = mp.Process(target=mcts_worker, args=(root, net, sims_per_worker, verbose))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    pi = create_pi_vector(root, tau)
    value = root.q
    chosen_action = int(torch.argmax(pi))
    return pi, value, chosen_action

# --- Example Main Block ---
if __name__ == '__main__':
    # Create an initial chess board.
    initial_state = chess.Board()
    
    # Dummy network: Replace with your actual PyTorch model.
    class DummyNet(torch.nn.Module):
        def forward(self, state_tensor):
            # For illustration, return a uniform policy and a zero value.
            policy = torch.ones(4672) / 4672
            value = torch.tensor(0.0)
            return policy, value

    net = DummyNet()
    
    # Run MCTS in parallel.
    tau = 1  # temperature parameter
    pi, value, chosen_action = mcts(initial_state, net, tau, total_sims=100, num_workers=4, verbose=True)
    print("Final policy vector (pi):", pi)
    print("Estimated value:", value)
    print("Chosen action index:", chosen_action)