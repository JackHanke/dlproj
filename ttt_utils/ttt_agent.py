import torch 
import chess
import chess.engine
from sys import platform
import time
from copy import deepcopy

# from utils.mcts import mcts
# from utils.mcts_parallel import mcts
from utils.chess_utils_local import get_action_mask, legal_moves, moves_to_actions
from utils.utils import renormalize_network_output

from ttt_utils.ttt_mcts import mcts
from ttt_utils.ttt_networks import DemoTicTacToeConvNet, DemoTicTacToeFeedForwardNet

class RandomAgent:
    def __init__(self):
        self.version = 0
        self.name = 'Random Agent'

    def inference(self, observation: list, board: list = None, device: torch.device = torch.device('cpu'), tau: float = 0):
        legal_action_indexes = torch.nonzero(torch.tensor(observation['action_mask']))
        random_index = torch.randint(low=0, high=legal_action_indexes.shape[0], size=(1,))
        action = legal_action_indexes[random_index].item()
        return action, 0

class Human:
    def __init__(self):
        self.version = 0
        self.name = 'Human Agent'

    def inference(self, observation: list, board: list = None, device: torch.device = torch.device('cpu'), tau: float = 0):
        diagram_str = '''
            0 3 6
            1 4 7
            2 5 8
        '''
        print(diagram_str)
        # TODO prevent type error ?
        action = int(input('Enter action > '))
        while observation['action_mask'][action] == 0:
            action = int(input('Please enter legal action > '))
        return action, 0

class PerfectAgent:
    def __init__(self, depth = 9):
        self.version = 0
        self.name = "Perfect Agent"
        self.depth = depth

    def inference(self, observation: torch.tensor, board: list = None, device: torch.device = torch.device('cpu'), tau: float = 1):

        def minimax_alpha_beta(board, depth, alpha, beta):
            
            def max_val(board, alpha, beta, depth_index):
                if board.check_game_over() or depth_index >= depth:
                    return evaluate(board=board), deepcopy(board)
                v = -float('inf')
                best_board = None
                # for board in game.generate_all_moves(board, PLAYER2_PIECE_COLOR):
                legal_moves = [i for i, mark in enumerate(board.squares) if mark == 0]
                for move in legal_moves:
                    copy_board = deepcopy(board)
                    copy_board.play_turn(agent = 0, pos = move)
                    new_val, _ = min_val(copy_board, alpha, beta, depth_index=depth_index+1)
                    if new_val > v:
                        best_board = copy_board
                        v = max(v, new_val)
                    if v >= beta: return v, best_board
                    alpha = max(alpha, v)
                return v, best_board

            def min_val(board, alpha, beta, depth_index):
                if board.check_game_over() or depth_index >= depth:
                    return evaluate(board=board), deepcopy(board)
                v = float('inf')
                best_board = None
                # for board in game.generate_all_moves(board, PLAYER1_PIECE_COLOR):
                legal_moves = [i for i, mark in enumerate(board.squares) if mark == 0]
                for move in legal_moves:
                    copy_board = deepcopy(board)
                    copy_board.play_turn(agent = 1, pos = move)
                    # input(new_board.squares)
                    new_val, _ = max_val(copy_board, alpha, beta, depth_index=depth_index+1)
                    if new_val < v:
                        best_board = copy_board
                        v = min(v, new_val)
                    if v <= alpha: return v, best_board
                    beta = min(beta, v)
                return v, best_board

            legal_moves = [i for i, mark in enumerate(board.squares) if mark == 0]
            if (len(legal_moves) % 2) == 0:
                score, best_board = min_val(board=board, alpha=alpha, beta=beta, depth_index=0)
            else:
                score, best_board = max_val(board=board, alpha=alpha, beta=beta, depth_index=0)

            for ind, (i, j) in enumerate(zip(board.squares, best_board.squares)):
                if i != j:
                    best_move = ind
            
            return best_move, score

        def evaluate(board):
            # NOTE -1 for no winner, 1 for agent 0 wins, 2 for agent 1 wins
            # agent 0 is who plays first, ie blue 'x'
            winner = board.check_for_winner()
            if winner == -1: return 0
            elif winner == 1: return 1
            elif winner == 2: return -1
            # return score

        action, value = minimax_alpha_beta(
            board=board, 
            depth=self.depth, 
            alpha=float('-inf'), 
            beta=float('inf')
        )

        return action, value

class Agent:
    def __init__(self, version: int, network: torch.nn.Module, sims: int = 1):
        self.version = version
        self.name = f'dem0 tictactoe version {self.version}'
        self.network = network
        self.sims = sims
        self.node_cache = None

    def inference(self, observation: torch.tensor, board: list = None, device: torch.device = torch.device('cpu'), tau: float = 0) -> tuple[int, float]:
        if self.sims > 1:
            _, value, action, _ = mcts(
                state=board, 
                observation=observation,
                net=self.network, 
                node=None,
                device=device, 
                tau=tau, 
                sims=self.sims, 
                inference_mode=True
            )
            return action, value

        # NOTE ! the following lines turn observation into a filtered network inference
        elif self.sims <= 1:
            observation_arr = observation['observation']
            action_mask_arr = observation['action_mask']
            obs = torch.tensor(observation_arr).float().permute(2,0,1).unsqueeze(0)
            self.network.eval()
            with torch.no_grad():
                policy, value = self.network.forward(obs)
                pi = renormalize_network_output(policy_vec=policy, legal_moves=torch.tensor(action_mask_arr))
            action = torch.argmax(pi).item()
            value = value.item()
            return action, value
    

