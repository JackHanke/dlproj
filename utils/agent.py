import torch 
import chess
import chess.engine
from utils.mcts import mcts
# from utils.mcts_parallel import mcts
from utils.chess_utils_local import get_action_mask, legal_moves, moves_to_actions
from sys import platform
import time

class Agent:
    def __init__(self, version: int, network: torch.nn.Module, sims: int = 10):
        self.version = version
        self.network = network
        self.sims = sims
        self.node_cache = None

    def inference(self, board_state: chess.Board, observation: torch.tensor, device: torch.device, tau: float = 0) -> tuple[int, float]:
        # get node cache
        root = None

        _, value, action, subtree_node = mcts(
            state=board_state, 
            observation=observation,
            net=self.network, 
            node=root,
            device=device, 
            tau=tau, 
            sims=self.sims, 
            inference_mode=True
        )

        # update node cache
        # self.node_cache = subtree_node

        return action, value
    
    def compute_bayes_elo(self, *args):
        raise NotImplementedError

class Stockfish:
    def __init__(self, level):
        # Sotckfish 5: https://stockfishchess.org/blog/2014/stockfish-5/
        self.version = "Stockfish"
        self.network = None
        self.sims = None
        self.nod_cache = None
        self.level = level
        if platform == 'darwin': stock_path = f'./stockfish-5-mac/Mac/stockfish-5-64'
        elif platform == 'linux': stock_path = f'./stockfish-5-mac/src/stockfish' 
        self.engine = chess.engine.SimpleEngine.popen_uci(stock_path)
        self.engine.configure({"Skill Level": self.level})

    def inference(self, board_state: chess.Board, observation: torch.tensor, device: torch.device, tau: float = 1):
        # fuck you PettingZoo
        if board_state.turn: # it white's turn
            uci_move = str(self.engine.play(board_state, chess.engine.Limit(time=1.0)).move)
        elif not board_state.turn: # if black's turn
            mirrored_board = board_state.mirror()
            uci_move = str(self.engine.play(mirrored_board, chess.engine.Limit(time=1.0)).move)
        _ = legal_moves(orig_board=board_state) # idk if this is necessary
        action = moves_to_actions[uci_move]
        value = 1 # NOTE compatability
        return action, value

    def compute_bayes_elo(self, *args):
        raise NotImplementedError
