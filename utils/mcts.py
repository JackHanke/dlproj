import torch
from torch.distributions import Categorical
import chess
import threading
import numpy as np
from copy import deepcopy
import gc

# --------------------------------------------------------------------------
# Replace these with your own utility imports or adapt them as needed
from utils.chess_utils_local import (
    get_observation,
    legal_moves,
    result_to_int,
    get_action_mask,
    moves_to_actions,
    mirror_move
)
from utils.utils import (
    prepare_state_for_net,
    filter_legal_moves_and_renomalize,
    observe
)

POSSIBLE_AGENTS = ['player_0', 'player_1']

# --------------------------------------------------------------------------
# Node class for MCTS
class Node:
    def __init__(
        self, 
        state: chess.Board, 
        board_history: np.ndarray, 
        agent: str, 
        parent=None, 
        prior: float = 0.0
    ):
        """
        state:         A python-chess Board object
        board_history: Your custom representation of past states
        agent:         Which agent (player_0 or player_1) acts from this state
        parent:        Parent Node (None if root)
        prior:         Prior probability p from the network
        """
        self.state = state
        self.board_history = board_history
        self.agent = agent
        self.parent = parent

        # Children stored in a dict keyed by chess.Move
        self.children = {}

        # MCTS statistics
        self.n = 0       # visit count
        self.w = 0.0     # total value
        self.q = 0.0     # mean value = w / n
        self.p = prior   # prior from network


def add_dirichlet_noise_to_root(root: Node, alpha=0.03, epsilon=0.25):
    legal_moves = list(root.children.keys())
    if not legal_moves:
        return

    noise = np.random.dirichlet([alpha] * len(legal_moves))
    for move, eta in zip(legal_moves, noise):
        child = root.children[move]
        child.p = (1 - epsilon) * child.p + epsilon * eta


def backup(node: Node, value: float):
    """
    Backup the value all the way up to the root.
    The sign of 'value' is flipped at each level (zero-sum).
    """
    v = value
    while node is not None:
        node.n += 1
        node.w += v
        node.q = node.w / node.n
        v = -v  # flip sign for parent
        node = node.parent


def is_game_over(board: chess.Board) -> bool:
    """
    Check if the chess game is over (no legal moves, insufficient material, etc.)
    """
    if not any(legal_moves(board)):
        return True
    if board.is_insufficient_material() or board.can_claim_draw():
        return True
    if board.is_game_over():
        return True
    return False


def create_pi_vector(node: Node, tau: float):
    """
    Create the final policy distribution (pi) from the node visit counts.
    For chess, you might have an action space of size 4672 (or whatever).
    """
    # Example uses 4672 as the total number of possible moves
    pi = [0.0] * 4672

    # If tau == 0, pick the move with highest visit count deterministically
    if tau == 0:
        most_visits = max(node.children.values(), key=lambda c: c.n) if node.children else None
        if most_visits is not None:
            # Find which action index corresponds to the best child
            for move, child in node.children.items():
                if child is most_visits:
                    # If it's black's turn, mirror the move
                    mirrored = mirror_move(move) if node.state.turn == chess.BLACK else move
                    action_idx = moves_to_actions[mirrored.uci()]
                    pi[action_idx] = 1.0
                    break
        return torch.tensor(pi)

    # If tau > 0, use a softmax-style distribution over n^(1/tau)
    sum_n_tau = sum([child.n ** (1.0 / tau) for child in node.children.values()])
    for move, child in node.children.items():
        mirrored = mirror_move(move) if node.state.turn == chess.BLACK else move
        action_idx = moves_to_actions[mirrored.uci()]
        pi[action_idx] = (child.n ** (1.0 / tau)) / (sum_n_tau + 1e-8)

    return torch.tensor(pi)

# --------------------------------------------------------------------------
# 1) Selection: find a leaf node by following UCB/PUCT from the root
def select_leaf_node(root: Node, c_puct: float = 2.5) -> Node:
    current = root
    while len(current.children) > 0:
        # Not a leaf, so pick the child with the best UCB score
        best_score = -float('inf')
        selected_move = None
        parent_visits = current.n + 1e-6

        for move, child in current.children.items():
            q = child.q
            p = child.p
            child_visits = child.n
            # PUCT formula
            u = c_puct * np.sqrt(parent_visits) * p / (1 + child_visits)
            score = q + u
            if score > best_score:
                best_score = score
                selected_move = move

        current = current.children[selected_move]
    # current is now a leaf (no children or game over)
    return current

# --------------------------------------------------------------------------
# 2) Batch expansion: gather leaf nodes, run one NN inference, expand each
@torch.no_grad()
def expand_nodes(leaf_nodes, net, device):
    """
    - leaf_nodes: list of Node objects that need expansion
    - net:        your PyTorch model returning (policy, value)
    - device:     the device (cuda/cpu)
    """
    if not leaf_nodes:
        return

    # Prepare a batched tensor of states
    states = []
    for node in leaf_nodes:
        # If it's terminal, we won't expand; just do a backup
        if is_game_over(node.state):
            states.append(None)
            continue
        # Otherwise, gather the board observation
        obs_dict = observe(
            board=node.state,
            agent=node.agent,
            possible_agents=POSSIBLE_AGENTS,
            board_history=node.board_history
        )
        obs = obs_dict['observation']
        states.append(obs)

    # Convert states to a batched tensor
    # Some nodes may be None (terminal), handle them carefully
    valid_indices = []
    batch_data = []
    for i, obs in enumerate(states):
        if obs is not None:
            # shape: [C, H, W] -> we’ll stack them along dim=0
            prep = prepare_state_for_net(obs.copy())  # returns shape [1, C, H, W]
            batch_data.append(prep)
            valid_indices.append(i)

    if len(batch_data) == 0:
        # All were terminal; just do backups for those
        for i, node in enumerate(leaf_nodes):
            if node is not None and is_game_over(node.state):
                backup(node, result_to_int(node.state.result(claim_draw=True)))
        return

    # Stack into a single batch
    batch_tensor = torch.cat(batch_data, dim=0).to(device)  # shape [batch_size, C, H, W]

    # Single forward pass
    policy_batch, value_batch = net(batch_tensor)
    # Suppose policy_batch.shape = [batch_size, action_space]
    #         value_batch.shape  = [batch_size, 1]

    # Distribute results back to each leaf node
    for b_idx, node_idx in enumerate(valid_indices):
        node = leaf_nodes[node_idx]

        # 2a) If game is over, skip expansion (already handled)
        if is_game_over(node.state):
            backup(node, result_to_int(node.state.result(claim_draw=True)))
            continue

        # 2b) Filter the policy with the legal moves
        action_mask = torch.tensor(get_action_mask(node.state), device=device)
        p_vec = filter_legal_moves_and_renomalize(policy_batch[b_idx], action_mask).squeeze(-1)

        # 2c) Expand children
        if len(node.children) == 0:
            for move, p_val in zip(node.state.legal_moves, p_vec.cpu()):
                next_state = deepcopy(node.state)
                next_state.push(move)
                # Example: shift board_history or however you handle it
                board_history = np.dstack((
                    get_observation(next_state, player=0)[:, :, 7:],
                    node.board_history[:, :, :-13]
                ))
                new_agent = POSSIBLE_AGENTS[0] if node.agent == POSSIBLE_AGENTS[1] else POSSIBLE_AGENTS[1]
                new_node = Node(
                    state=next_state,
                    board_history=board_history,
                    agent=new_agent,
                    parent=node,
                    prior=p_val.item()
                )
                node.children[move] = new_node

        # 2d) Backup the values
        val = value_batch[b_idx].item()
        backup(node, val)

# --------------------------------------------------------------------------
# 3) Main MCTS loop with batched inference
@torch.no_grad()
def mcts(
    state: chess.Board,
    starting_agent: str,
    net: torch.nn.Module,
    device: torch.device,
    sims: int = 100,
    batch_size: int = 8,
    c_puct: float = 2.5,
    tau: float = 1.0,
    inference_mode: bool = False
):
    """
    state:         Initial chess.Board
    starting_agent: 'player_0' or 'player_1'
    net:           Your PyTorch model returning (policy, value)
    device:        torch device (cpu or cuda)
    sims:          Total MCTS simulations
    batch_size:    How many leaf nodes to collect before one inference
    c_puct:        Exploration constant
    tau:           Temperature for final policy
    inference_mode: If True, pick argmax instead of sampling from pi
    """
    net.to(device)
    net.eval()

    # Create the root node
    board_history = np.zeros((8, 8, 104), dtype=bool)
    root = Node(
        state=state,
        board_history=board_history,
        agent=starting_agent
    )

    simulations_done = 0
    leaf_nodes = []
    root_expanded = False  # track whether we've expanded root yet

    while simulations_done < sims:
        # 3a) Selection
        leaf = select_leaf_node(root, c_puct=c_puct)
        if is_game_over(leaf.state):
            backup(leaf, result_to_int(leaf.state.result(claim_draw=True)))
            simulations_done += 1
            continue

        # 3b) Queue for expansion
        leaf_nodes.append(leaf)
        simulations_done += 1

        # 3c) Run expansion if batch full or done simulating
        if len(leaf_nodes) >= batch_size or simulations_done == sims:
            expand_nodes(leaf_nodes, net, device)

            # Add Dirichlet noise to root node after first expansion
            if not root_expanded:
                add_dirichlet_noise_to_root(root)
                root_expanded = True

            leaf_nodes = []

    # After all sims, construct final policy and value
    pi = create_pi_vector(root, tau)
    value = root.q

    if inference_mode:
        sampled_action = int(pi.argmax().item())
    else:
        m = Categorical(pi)
        sampled_action = int(m.sample().item())

    return pi, value, sampled_action, root
