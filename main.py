import threading
import torch
from utils.training import train_on_batch
from utils.self_play import SelfPlayAgent
from utils.memory import ReplayMemory
from utils.networks import DemoNet
from utils.configs import load_config
from utils.optimizer import get_optimizer  


def setup():
    """Initialize configurations, network, replay buffer, and optimizer."""
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize replay buffer
    replay_buffer = ReplayMemory(maxlen=config.training.data_buffer_size)

    # Initialize network
    network = DemoNet(num_res_blocks=config.network.num_residual_blocks).to(device)

    # Get optimizer using the helper function
    optimizer = get_optimizer(
        optimizer_name=config.training.optimizer.lower(),
        lr=config.training.learning_rate.initial,
        weight_decay=config.training.weight_decay,
        model=network,
        momentum=config.training.momentum  # Only applies if using SGD
    )

    # Initialize self-play agent
    self_play_agent = SelfPlayAgent(v_resign_start=config.self_play.resign_threshold)

    return config, device, replay_buffer, network, optimizer, self_play_agent


def self_play_loop(self_play_agent, replay_buffer, network, config):
    """Continuously generates self-play games and updates the replay buffer."""
    while True:
        self_play_agent.run_self_play(
            training_data=replay_buffer,
            network=network,
            n_sims=config.self_play.num_simulations,
            num_games=config.training.num_self_play_games,
            max_moves=200,  # Adjustable max moves per game
            max_replay_len=replay_buffer.maxlen,
            epsilon=config.self_play.exploration_noise.epsilon,
            alpha=config.self_play.exploration_noise.alpha
        )


def training_loop(replay_buffer, network, optimizer, config, device):
    """Continuously trains the network using replay buffer data."""
    while True:
        train_on_batch(
            data=replay_buffer,
            network=network,
            batch_size=config.training.batch_size,
            device=device,
            optimizer=optimizer
        )


def main():
    """Main function to initialize components and start self-play and training in parallel."""
    config, device, replay_buffer, network, optimizer, self_play_agent = setup()

    # Create self-play and training threads
    self_play_thread = threading.Thread(
        target=self_play_loop, 
        args=(self_play_agent, replay_buffer, network, config), 
        daemon=True
    )
    training_thread = threading.Thread(
        target=training_loop, 
        args=(replay_buffer, network, optimizer, config, device), 
        daemon=True
    )

    # Start threads
    self_play_thread.start()
    training_thread.start()

    # Ensure threads run indefinitely
    self_play_thread.join()
    training_thread.join()


if __name__ == "__main__":
    main()