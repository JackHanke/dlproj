import torch
import time
import multiprocessing as mp
from copy import deepcopy

from utils.training import train_on_batch
from utils.self_play import SelfPlaySession
from utils.memory import ReplayMemory
from utils.networks import DemoNet
from utils.optimizer import get_optimizer  
from utils.evaluator import evaluator
from utils.agent import Agent


def training_loop(stop_event, memory, network, device, optimizer_params):
    """
    Continuously train on a batch until stop_event is set.
    The optimizer is created inside the child process using optimizer_params.
    """
    # Create the optimizer instance in the child process
    optimizer = get_optimizer(
        optimizer_name=optimizer_params["optimizer_name"],
        lr=optimizer_params["lr"],
        weight_decay=optimizer_params["weight_decay"],
        model=network,
        momentum=optimizer_params.get("momentum", 0.0)
    )
    
    i = 0
    while not stop_event.is_set():
        if i < 2:
            print('This is a test to show multiprocessing works.')
        train_on_batch(
            data=memory,
            network=network,
            batch_size=1,
            device=device,
            optimizer=optimizer
        )
        i += 1
        time.sleep(0.01)


def main():
    device = "cpu"
    self_play_session = SelfPlaySession()
    memory = ReplayMemory(1000)
    
    # Create networks and share their memory
    current_best_network = DemoNet(num_res_blocks=1)
    challenger_network = DemoNet(num_res_blocks=1)
    challenger_network.share_memory()  # Ensure the model's parameters are in shared memory
    
    # Loop twice for this example
    for i in range(2):
        print(">" * 50)
        print(f'Dem0 Iteration {i+1}:\n')
        
        # Initialize agents
        current_best_agent = Agent(version=0, network=current_best_network, sims=5)
        challenger_agent = Agent(version=1, network=challenger_network, sims=5)
        
        # Instead of creating an optimizer instance here,
        # we prepare parameters for its creation.
        optimizer_params = {
            "optimizer_name": "adam",
            "lr": 0.0001,
            "weight_decay": 1e-4,
            "momentum": 0.9  # Only used if you switch to SGD
        }
        
        # Create a stop event and start the training process
        stop_event = mp.Event()
        training_process = mp.Process(
            target=training_loop,
            args=(stop_event, memory, challenger_network, device, optimizer_params)
        )
        training_process.start()
        
        # Run self-play concurrently with training
        print('Running self play...')
        self_play_session.run_self_play(
            training_data=memory,
            network=challenger_network,
            n_sims=2,
            num_games=2,
            max_moves=500
        )
        
        # Self-play finished, signal the training process to stop
        stop_event.set()
        training_process.join()  # Wait for the training process to exit
        
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
        
        # Update networks for the next iteration
        current_best_network = deepcopy(current_best_agent.network)
        challenger_network = current_best_agent.network
        challenger_network.share_memory()  # Make sure to share memory for the updated network

if __name__ == "__main__":
    mp.set_start_method("spawn")  # Ensures compatibility across platforms
    main()