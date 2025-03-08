import numpy as np
from copy import deepcopy
from utils.chess_utils_local import get_observation, legal_moves, result_to_int, get_action_mask
from utils.utils import prepare_state_for_net, filter_legal_moves, filter_legal_moves_and_renomalize, renormalize_network_output, rand_argmax
import torch
import time
import chess

import concurrent.futures
# import torch.multiprocessing as mp
# import torch.multiprocessing.managers
import multiprocessing
from multiprocessing.managers import BaseManager


# NOTE this code refers to a chess.Move and a UCI string as a 'move'
# while the action indexes provided by Pettingzoo as an 'action'

# node of Monte Carlo Tree


# creates the final pi tensor (same size as network output)
def create_pi_vector(node, tau: float):
    pi = [0 for _ in range(4672)]
    pi_denom = sum([child.n**(1/tau) for child in node.children.values()])

    action_mask = torch.tensor(get_action_mask(orig_board=node.state))
    legal_action_indexes = torch.nonzero(action_mask)

    legal_moves_uci = [move_obj.uci() for move_obj in node.state.legal_moves]

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
def choose_move(node, net: torch.nn.Module, recursion_count: int, verbose: bool = False):
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
    print(f'net evaluation: {time.time()- start} s')

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
# TODO safe to remove?
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
def backup(node, v: int = 0):
    node.n += 1
    node.w += v
    node.q = node.w/node.n
    # 
    if node.parent is None: return
    # TODO what is virtual loss?
    return backup(node=node.parent, v=v)

@torch.no_grad()
def expand(node, recursion_count: int = 0, verbose: bool = False):
    expand_time = time.time()
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
        choose_time = time.time()
        chosen_move = choose_move(node=node, recursion_count=recursion_count, verbose=verbose)
        choose_time = time.time()-choose_time
        print(f'Time to choose: {choose_time}')
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
        input(f'Time to expand: {choose_time/(time.time()-expand_time) * 100 :.4f} %')
        return expand(node=next_node, recursion_count=recursion_count+1, verbose=verbose)

# TODO helper for parallelization attempts
def helper(x):
    sims = x[2]
    for sim in range(sims):
        expand(node=x[0], net=x[1])
    return x[0].n


# TODO change this stuff

# Function to simulate the process creating data
def data_producer(queue, process_id):
    for _ in range(5):  # Simulating 5 data points per process
        data = np.random.rand(10)  # 10 features per data point
        queue.put((process_id, data))  # Put data with process_id for later identification

# Function to collect and batch data, and forward to NN
def process_batch(queue, result_queue, batch_size=10):
    batch_data = []
    process_ids = []
    
    while True:
        # Collect data from processes
        process_id, data = queue.get()
        batch_data.append(data)
        process_ids.append(process_id)

        # If the batch size is reached, process the batch
        if len(batch_data) >= batch_size:
            batch_data_tensor = torch.tensor(batch_data, dtype=torch.float32)
            model = SimpleNN()  # Example: load a model
            outputs = model(batch_data_tensor)

            # Send the results back to the processes
            for idx, process_id in enumerate(process_ids):
                result_queue.put((process_id, outputs[idx].item()))
            
            # Reset for the next batch
            batch_data = []
            process_ids = []

        # Exit condition (could be based on some criteria, like a specific number of batches)
        if len(batch_data) == 0:
            break

# Function to handle result forwarding
def result_receiver(result_queue):
    while True:
        process_id, result = result_queue.get()
        print(f"Process {process_id} received result: {result}")
        # In a real scenario, you'd forward the result back to the process


# conduct Monte Carlo Tree Search for sims sims and the python-chess state
# using network net and temperature tau
def mcts(state: chess.Board, net: torch.nn.Module, tau: int, sims: int = 1, verbose: bool = False):
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

    BaseManager().register('Node', Node)
    manager = BaseManager()
    manager.start()
    root = manager.Node(state=state)

    processes = []
    for i in range(sims):
        process = multiprocessing.Process(target=expand, args=(root, 0))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()

    # # Turn data from tree into pi vector
    pi = create_pi_vector(node=root._getvalue(), tau=tau)
    # value is the mean value from the root
    value = root.q
    # get best value calculated from pi
    chosen_action = int(torch.argmax(pi))
    return pi, value, chosen_action

    # with BaseManager() as manager:
    #     manager.register('Node', Node)

    #     root = manager.Node(state=state)


    #     processes = []
    #     for i in range(5):
    #         process = multiprocessing.Process(target=add_item, args=(shared_object, i))
    #         processes.append(process)
    #         process.start()

    #     # Wait for all processes to finish
    #     for process in processes:
    #         process.join()
    # 
    # tot_start = time.time()
    # # number of processes
    # num_procs = 4
    # # number of sims per tree
    # by_tree_sims = sims // num_procs

    # # multiprocessing queues for data collection and result distribution
    # queue = mp.Queue()
    # result_queue = mp.Queue()

    # # Start the data producers (multiple processes)
    # processes = []
    # for i in range(4):  # 4 processes generating data
    #     p = mp.Process(target=data_producer, args=(queue, i))
    #     processes.append(p)
    #     p.start()

    # # Start the batch processing and result forwarding
    # batch_process = mp.Process(target=process_batch, args=(queue, result_queue))
    # batch_process.start()

    # # Start the result receiver (to handle responses)
    # result_process = mp.Process(target=result_receiver, args=(result_queue,))
    # result_process.start()

    # # Wait for processes to complete
    # for p in processes:
    #     p.join()

    # # Let the batch process know that it's time to stop (this is just an example)
    # batch_process.join()

    # tot_start = time.time()
    # processes = []
    # for _ in range(num_procs):
    #     # p = mp.Process(target=expand, args=[deepcopy(root), net])
    #     p = mp.Process(target=helper, args=[(deepcopy(root), net, by_tree_sims)])
    #     p.start()
    #     processes.append(p)

    # for process in processes:
    #     process.join()

    # if verbose: print(f'Total time for async {sims} sims: {time.time()-tot_start} s')
    
    # # # Turn data from tree into pi vector
    # pi = create_pi_vector(node=root, tau=tau)
    # # value is the mean value from the root
    # value = root.q
    # # get best value calculated from pi
    # chosen_action = int(torch.argmax(pi))
    # return pi, value, chosen_action



    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     trees = [(deepcopy(root), net, by_tree_sims) for _ in range(num_procs)]
    #     results = executor.map(helper, trees)
    #     for result in results:
    #         print(result)

    #     # results = [executor.submit(helper, (deepcopy(root), net, by_tree_sims)) for _ in range(num_procs)]
    #     # for f in concurrent.futures.as_completed(results):
    #     #     print(f.result())

    # if verbose: print(f'Total time for async {sims} sims: {time.time()-tot_start} s')

    # # TODO we are doing root parallelization, so we need a function to combine two trees
    # # TODO batching net inference between all procs, AND sending it back to all 
    # # 
