import torch
import time
import torch.multiprocessing as mp
from copy import deepcopy

from utils.training import train_on_batch, Checkpoint
from utils.self_play import SelfPlaySession
from utils.memory import ReplayMemory
from utils.networks import DemoNet
from utils.optimizer import get_optimizer  
from utils.evaluator import evaluator
from utils.agent import Agent
from utils.utils import Timer
import os


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
    print(f'Memory length after self play: {len(memory)}')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self_play_session = SelfPlaySession()
    memory = ReplayMemory(1000)
    checkpoint = Checkpoint(verbose=True, compute_elo=False)
    
    current_best_network = DemoNet(num_res_blocks=1)
    challenger_network = deepcopy(current_best_network)
    base_path = "checkpoints/best_model/"
    weights_path = os.path.join(base_path, "weights.pth")
    info_path = os.path.join(base_path, "info.json")

    current_best_version = 0
    for i in range(2):
        print(">" * 50)
        print(f'Dem0 Iteration {i+1}:\n')
        
        current_best_agent = Agent(version=current_best_version, network=current_best_network, sims=5)
        challenger_agent = Agent(version=current_best_version+1, network=challenger_network, sims=5)
        
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
        
        print('Running self play...')
        self_play_session.run_self_play(
            training_data=memory,
            network=current_best_network,
            n_sims=2,
            num_games=2,
            max_moves=500
        )
        
        stop_event.set()
        training_process.join()
        
        print(f"\nTraining iterations completed: {counter.value}")  # Print the value of i

        print("\nEvaluating...")
        current_best_agent = evaluator(
            challenger_agent=challenger_agent, 
            current_best_agent=current_best_agent, 
            max_moves=500,
            num_games=3, 
            v_resign=self_play_session.v_resign, 
            verbose=True
        )
        print(f'After this loop, the best_agent is {current_best_agent.version}\n\n')

        current_best_network = deepcopy(current_best_agent.network)
        challenger_network = current_best_agent.network
        current_best_version = current_best_agent.version

        # step checkpoint
        checkpoint.step(weights_path=weights_path, info_path=info_path, current_best_agent=deepcopy(current_best_agent))

if __name__ == "__main__":
    with Timer():
        mp.set_start_method("spawn")  
        main()