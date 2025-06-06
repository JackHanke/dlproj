import torch
import torch.multiprocessing as mp
import time
import os
from tqdm import tqdm
from copy import deepcopy
import logging
from datetime import datetime
import matplotlib.pyplot as plt

from utils.training import train_on_batch, Checkpoint
from utils.memory import ReplayMemory
from utils.optimizer import get_optimizer  
from utils.utils import Timer

from ttt_utils.ttt_self_play import SelfPlaySession
from ttt_utils.ttt_agent import Agent, RandomAgent, PerfectAgent, Human
from ttt_utils.ttt_evaluator import evaluator
from ttt_utils.ttt_networks import DemoTicTacToeConvNet, DemoTicTacToeFeedForwardNet


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
            batch_size=32,
            device=device,
            optimizer=optimizer
        )
        i += 1

    counter.value = i  # Store the final value of i in the shared variable
    logging.info(f'Memory length after self play: {len(memory)}')

# mp_training is whether or not to use multiprocessing training to run while self-play runs
def main(mp_training: bool):
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='ttt_dem0.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f'\n\nRunning Experiment on {datetime.now()} with the following configs:')
    # TODO add all configs to logging for this log

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self_play_session = SelfPlaySession()
    memory = ReplayMemory(2000)
    training_epochs_per_iter = 10
    optimizer_params = {
        "optimizer_name": "adam",
        "lr": 0.0001,
        "weight_decay": 1e-4,
        "momentum": 0.9  
    }
        
    checkpoint = Checkpoint(verbose=True, compute_elo=False)
    
    # current_best_network = DemoTicTacToeConvNet(
    #     num_res_blocks=1
    # ).to(device)
    current_best_network = DemoTicTacToeFeedForwardNet(
        num_layers=3
    ).to(device)

    challenger_network = deepcopy(current_best_network).to(device)

    base_path = "checkpoints/best_model/"
    weights_path = os.path.join(base_path, "weights.pth")
    info_path = os.path.join(base_path, "info.json")
    model_path = os.path.join(base_path, "model.pth")

    current_best_version = 0
    i = 0 # number of iterations performed
    while True:
        iter_start_time = time.time()
        # print(">" * 50)
        # print()
        logger.info(f'Beginning dem0 Iteration {i+1}...\n')
        # TODO do we have to remake the agent every iter?
        current_best_agent = Agent(
            version=current_best_version, 
            network=current_best_network, 
            sims=15
        )
        challenger_agent = Agent(
            version=current_best_version+1, 
            network=challenger_network, 
            sims=15
        )

        # 
        self_play_session.run_self_play(
            training_data=memory,
            network=current_best_network,
            device=device,
            n_sims=15,
            num_games=50,
            max_moves=10
        )

        optimizer = get_optimizer(
            optimizer_name=optimizer_params["optimizer_name"],
            lr=optimizer_params["lr"],
            weight_decay=optimizer_params["weight_decay"],
            model=challenger_network,
            momentum=optimizer_params.get("momentum", 0.0)
        )
        # NOTE still uses multiprocessing queue, see if this changes anything
        train_bar = tqdm(range(1, training_epochs_per_iter+1))
        for train_idx in train_bar:
            train_on_batch(
                data=memory,
                network=challenger_network,
                batch_size=32,
                device=device,
                optimizer=optimizer
            )
            train_bar.set_description(f"Training, Epoch {train_idx}.")

        
        # print(f"\nTraining iterations completed: {counter.value}")  # Print the value of i
        # logger.debug(f'Training iterations completed: {counter.value}')

        # print("\nEvaluating...")
        new_best_agent, wins, draws, losses, win_percent, tot_games = evaluator(
            challenger_agent=challenger_agent, 
            current_best_agent=current_best_agent,
            device=device,
            max_moves=10,
            num_games=50,
            v_resign=self_play_session.v_resign
        )
        # print(f'After this loop, the best_agent is {current_best_agent.version}\n\n')
        logger.info(f'Agent {challenger_agent.version} playing Agent {current_best_agent.version}, won {wins} games, drew {draws} games, lost {losses} games. ({round(100*win_percent, 2)}% wins, {round(100*(draws/tot_games), 2)}% draws))')

        current_best_network = deepcopy(new_best_agent.network).to(device)
        challenger_network = new_best_agent.network.to(device)
        current_best_version = new_best_agent.version

        # print("\nExternal evaluating...")
        perfect_agent = PerfectAgent()
        wins, draws, losses, win_percent, tot_games = evaluator(
            challenger_agent=current_best_agent,
            current_best_agent=perfect_agent,
            device=device,
            max_moves=10,
            num_games=50,
            v_resign=self_play_session.v_resign,
            external = True
        )

        logger.info(f'Against Perfect Agent, won {wins} games, drew {draws} games, lost {losses} games. ({round(100*win_percent, 2)}% wins, {round(100*(draws/tot_games), 2)}% draws)')
        
        # offset = 0
        # for level, progress in stockfish_progress.items():
        #     # plt.plot([i+offset for i in range(len(progress))], progress, label=f'Stockfish Level {level}')
        #     offset += len(progress)

            # plt.ion()
            # self_play_games = 10 # TODO change this to config var
            # plt.xlabel(f'Training Loops ({self_play_games} games per)')
            # plt.ylabel(f'Win Percentage')
            # plt.title('dem0 Training Against Stockfish Levels')
            # plt.show(block=False)

        # step checkpoint
        checkpoint.step(
            weights_path=weights_path, 
            model_path=model_path, 
            info_path=info_path, 
            current_best_agent=deepcopy(current_best_agent)
        )

        i += 1
        logger.info(f'Full iteration completed in {round(time.time()-iter_start_time, 2)} s.')

if __name__ == "__main__":
    with Timer():
        mp.set_start_method("spawn")  
        main(mp_training=False)
