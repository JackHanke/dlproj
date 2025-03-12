import torch 
import chess
import chess.engine
from utils.mcts import mcts
# from utils.mcts_parallel import mcts
from utils.chess_utils_local import get_action_mask, legal_moves
from sys import platform

class Agent:
    def __init__(self, version: int, network: torch.nn.Module, sims: int = 10):
        self.version = version
        self.network = network
        self.sims = sims
        self.node_cache = None

    def inference(self, board_state: chess.Board, device: torch.device, tau: float = 0) -> tuple[int, float]:
        _, value, action = mcts(
            state=board_state, 
            net=self.network, 
            device=device, 
            tau=tau, 
            sims=self.sims, 
            inference_mode=True
        )
            # given_node=self.node_cache
        # self.node_cache = child_node
        # TODO once this returns a node as well, we cache the tree with the agent
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

    def inference(self, board_state: chess.Board, device: torch.device, tau: float = 1):
        uci_move = self.engine.play(board_state, chess.engine.Limit(time=1.0)).move
        # NOTE translate uci_move to PettingZoo action number
        for ind, move in enumerate(board_state.legal_moves):
            if str(move.uci()) == str(uci_move):
                i = ind
                break
        action = legal_moves(orig_board=board_state)[i]

        value = 1 # NOTE compatability
        return action, value

    def compute_bayes_elo(self, *args):
        raise NotImplementedError