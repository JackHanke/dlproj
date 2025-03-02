import threading
import torch
from utils.training import train_on_batch
from utils.self_play import SelfPlaySession
from utils.memory import ReplayMemory
from utils.networks import DemoNet
from utils.configs import load_config, Config
from utils.optimizer import get_optimizer  
from utils.evaluator import evaluator
from utils.agent import Agent
from copy import deepcopy


def setup():
    """Initialize configurations, network, replay buffer, and optimizer."""
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize replay buffer
    replay_buffer = ReplayMemory(maxlen=config.training.data_buffer_size)

    # Initialize network
    network = DemoNet(num_res_blocks=1).to(device)

    # Get optimizer using the helper function
    optimizer = get_optimizer(
        optimizer_name=config.training.optimizer,
        lr=config.training.learning_rate.initial,
        weight_decay=config.training.weight_decay,
        model=network,
        momentum=config.training.momentum  # Only applies if using SGD
    )

    # Initialize self-play agent
    self_play_agent = SelfPlaySession(v_resign_start=config.self_play.resign_threshold)

    return config, device, replay_buffer, network, optimizer, self_play_agent


def self_play(self_play_session: SelfPlaySession, replay_buffer: ReplayMemory, network: DemoNet, config: Config):
    """Run a self-play session and store data in the replay buffer."""
    self_play_session.run_self_play(
        training_data=replay_buffer,
        network=network,
        n_sims=5,
        num_games=config.training.num_self_play_games,
        max_moves=200,  # Adjustable max moves per game
        max_replay_len=config.training.data_buffer_size,
        epsilon=config.self_play.exploration_noise.epsilon,
        alpha=config.self_play.exploration_noise.alpha
    )


def train(replay_buffer: ReplayMemory, network: DemoNet, optimizer: any, config: Config, device: torch.device, self_play_thread: threading.Thread):
    """Continuously train while self-play is running."""
    print("Training started.")

    while self_play_thread.is_alive():  # Keep training while self-play is active
        train_on_batch(
            data=replay_buffer,
            network=network,
            batch_size=config.training.batch_size,
            device=device,
            optimizer=optimizer
        )

    print("Training stopped because self-play is complete.")


def main():
    device = "cpu"
    self_play_session = SelfPlaySession()
    memory = ReplayMemory(1000)
    current_best_network = DemoNet(num_res_blocks=1)
    challenger_network = DemoNet(num_res_blocks=1)
    # Suppose we only loop twice for this example
    for i in range(2):
        print(">"*50, f'Dem0 Iteration {i+1}:','\n\n' ,sep='\n')
        # Init networks
        current_best_agent = Agent(version=0, network=current_best_network, sims=5)
        challenger_agent = Agent(version=1, network=challenger_network, sims=5)

        # run self play on challenger network
        print('Running self play...')
        self_play_session.run_self_play(
            training_data=memory,
            network=challenger_network,
            n_sims=2,
            num_games=2,
            max_moves=500
        )

        print("\nTraining on batch...")
        train_on_batch(
            data=memory, 
            network=challenger_network,
            batch_size=1,
            device=device,
            optimizer=get_optimizer('adam', lr=0.0001, weight_decay=1e-4, model=current_best_network)
        )

        print("\nEvaluating...")
        current_best_agent = evaluator(
            challenger_agent=challenger_agent, 
            current_best_agent=current_best_agent, 
            max_moves=500,
            num_games=3, 
            v_resign=self_play_session.v_resign, 
            verbose=True
        )
        print(f'After this loop, the best_agent is {current_best_agent.version}')
        print('\n\n')

        # Save best without allowing it to update next iteration
        current_best_network = deepcopy(current_best_agent.network)
        # Link to network
        challenger_network = current_best_agent.network


if __name__ == "__main__":
    main()