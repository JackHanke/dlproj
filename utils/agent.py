import torch 
from utils.mcts import mcts
import chess

class Agent:
    def __init__(self, version: int, network: torch.nn.Module, sims: int = 10):
        self.version = version
        self.network = network
        self.sims = sims
        self.node_cache = None

    def inference(self, board_state: chess.Board, tau: float = 1):
        _, value, action = mcts(state=board_state, net=self.network, tau=tau, sims=self.sims)
        # TODO once this returns a node as well, we cache the tree with the agent
        return action, value



