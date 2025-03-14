import torch
import pickle
from utils.agent import Agent, Stockfish
from utils.evaluator import evaluator
from utils.networks import DemoNet

# NOTE this script baselines the pretrained network against the version it is trained off of

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get pretrained network
    pretrained_net = DemoNet(num_res_blocks=10)
    pretrained_net.load_state_dict(torch.load(f'tests/pretrained_model.pth', weights_only=True))
    pretrained_net.eval()

    # NOTE training for sims=2 is equivalent to not using MCTS, can be used to further baseline
    pretrained_agent = Agent(
        version = 0, 
        network=pretrained_net,
        sims=100
    )
    stockfish_level = 0
    stockfish = Stockfish(level=stockfish_level, move_time=0.2)
    wins, draws, losses, win_percent, tot_games = evaluator(
        challenger_agent=pretrained_agent,
        current_best_agent=stockfish,
        device=device,
        max_moves=250,
        num_games=20,
        v_resign=-0.95
    )
    
    print(f'Against Stockfish 5 Level {stockfish_level}, {pretrained_agent.name}  won {wins} games, drew {draws} games, lost {losses} games. ({round(100*win_percent, 2)}% wins.)')

