import torch
import torch.multiprocessing as mp
import time
import os
from copy import deepcopy
import logging
from datetime import datetime
import matplotlib.pyplot as plt

from utils.training import train_on_batch, Checkpoint
from utils.self_play import SelfPlaySession
from utils.memory import ReplayMemory
from utils.networks import DemoNet
from utils.optimizer import get_optimizer  
from utils.evaluator import evaluator
from utils.agent import Agent, Stockfish
from utils.utils import Timer


def training_loop(stop_event, memory, network, device, optimizer_params, counter):
    """
    Continuously train on a batch until stop_event is set.
    The optimizer is created inside the child process using optimizer_params.
    """
    optimizer = get_optimizer(
        optimizer_name=optimizer_params["optimizer_name"],
        lr=optimizer_params["lr"],
        weight_decay=optimizer_params["weight_decay"],
        model=network,
        momentum=optimizer_params.get("momentum", 0.0)
    )
    
    i = 0
    while not stop_event.is_set():
        train_on_batch(
            data=memory,
            network=network,
            batch_size=1,
            device=device,
            optimizer=optimizer
        )
        i += 1

    counter.value = i  # Store the final value of i in the shared variable
    logging.debug(f'Memory length after self play: {len(memory)}')


def main():
    logger = logging.getLogger(__name__)

    # formatter = logging.Formatter("")
    logging.basicConfig(filename='dem0.log', level=logging.DEBUG, format='%(asctime)s::%(levelname)s:%(message)s')
    
    logging.info(f'\n\nRunning Experiment on {datetime.now()} with the following configs:')
    # TODO add all configs to logging for this log

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self_play_session = SelfPlaySession()
    memory = ReplayMemory(1000)
    checkpoint = Checkpoint(verbose=True, compute_elo=False)
    
    current_best_network = DemoNet(num_res_blocks=1).to(device)
    challenger_network = deepcopy(current_best_network).to(device)
    base_path = "checkpoints/best_model/"
    weights_path = os.path.join(base_path, "weights.pth")
    info_path = os.path.join(base_path, "info.json")

    stockfish_level = 0
    stockfish_progress = {0:[]}
    current_best_version = 0
    i = 0 # number of iterations performed
    while True:
        # print(">" * 50)
        # print()
        logger.debug(f'dem0 Iteration {i+1}:\n')
        
        current_best_agent = Agent(
            version=current_best_version, 
            network=current_best_network, 
            sims=100
        )
        challenger_agent = Agent(
            version=current_best_version+1, 
            network=challenger_network, 
            sims=100
        )
        
        optimizer_params = {
            "optimizer_name": "adam",
            "lr": 0.0001,
            "weight_decay": 1e-4,
            "momentum": 0.9  
        }
        
        stop_event = mp.Event()
        counter = mp.Value("i", 0)  # Shared integer for storing i
        
        training_process = mp.Process(
            target=training_loop,
            args=(stop_event, memory, challenger_network, device, optimizer_params, counter)
        )
        training_process.start()
        
        # print('Running self play...')
        self_play_session.run_self_play(
            training_data=memory,
            network=current_best_network,
            device=device,
            n_sims=100,
            num_games=1,
            max_moves=100
        )
        
        stop_event.set()
        training_process.join()
        
        # print(f"\nTraining iterations completed: {counter.value}")  # Print the value of i
        logger.debug(f'Training iterations completed: {counter.value}')

        # print("\nEvaluating...")
        current_best_agent = evaluator(
            challenger_agent=challenger_agent, 
            current_best_agent=current_best_agent,
            device=device,
            max_moves=100,
            num_games=10,
            v_resign=self_play_session.v_resign
        )
        # print(f'After this loop, the best_agent is {current_best_agent.version}\n\n')
        logger.debug(f'After this loop, the best_agent is {current_best_agent.version}')

        current_best_network = deepcopy(current_best_agent.network).to(device)
        challenger_network = current_best_agent.network.to(device)
        current_best_version = current_best_agent.version

        # print("\nExternal evaluating...")
        stockfish = Stockfish(level=stockfish_level)
        win_percent, loss_percent = evaluator(
            challenger_agent=current_best_agent,
            current_best_agent=stockfish,
            device=device,
            max_moves=100,
            num_games=10,
            v_resign=self_play_session.v_resign
        )
        # print(f'Against Stockfish 5 Level {stockfish_level}, won {win_percent} games, lost {loss_percent} games.')
        logger.debug(f'Against Stockfish 5 Level {stockfish_level}, won {win_percent} games, lost {loss_percent} games.')
        
        # logging
        stockfish_progress[stockfish_level].append(win_percent)

        if win_percent > 0.55:
            # update stockfish to higher level
            stockfish_level += 1
            stockfish.engine.configure({"Skill Level": stockfish_level})
            # print(f'! Upgrading Stockfish level to {stockfish_level}.')
            logger.info(f'! Upgrading Stockfish level to {stockfish_level}.')

            offset = 0
            for level, progress in stockfish_progress.items():
                plt.plot([i+offset for i in range(len(progress))], progress, label=f'Stockfish Level {level}')
                offset += len(progress)

            self_play_games = 10 # TODO change this to config var
            plt.xlabel(f'Training Loops ({self_play_games} games per)')
            plt.ylabel(f'Win Percentage')
            plt.title('dem0 Training Against Stockfish Levels')
            plt.show()

        # step checkpoint
        checkpoint.step(weights_path=weights_path, info_path=info_path, current_best_agent=deepcopy(current_best_agent))
    
    stockfish.engine.close()

if __name__ == "__main__":
    with Timer():
        mp.set_start_method("spawn")  
        main()
