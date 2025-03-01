import threading
import torch
from utils.training import train_on_batch
from utils.self_play import SelfPlayAgent
from utils.memory import ReplayMemory
from utils.networks import DemoNet
from utils.configs import load_config, Config
from utils.optimizer import get_optimizer  


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
    self_play_agent = SelfPlayAgent(v_resign_start=config.self_play.resign_threshold)

    return config, device, replay_buffer, network, optimizer, self_play_agent


def self_play(self_play_agent: SelfPlayAgent, replay_buffer: ReplayMemory, network: DemoNet, config: Config):
    """Run a self-play session and store data in the replay buffer."""
    self_play_agent.run_self_play(
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
    """Main function to initialize components and start self-play and training in parallel."""
    config, device, replay_buffer, network, optimizer, self_play_agent = setup()

    for i in range(1000):  # Run for 1000 iterations
        print(f"\nStarting iteration {i+1}")

        # Create the self-play thread
        self_play_thread = threading.Thread(
            target=self_play, 
            args=(self_play_agent, replay_buffer, network, config), 
            daemon=True
        )

        # Start the self-play thread
        self_play_thread.start()

        # Create and start the training thread (keeps training while self-play is running)
        training_thread = threading.Thread(
            target=train, 
            args=(replay_buffer, network, optimizer, config, device, self_play_thread), 
            daemon=True
        )

        training_thread.start()

        # Wait for self-play to complete
        self_play_thread.join()

        # Wait for training to finish (it will stop once self-play ends)
        training_thread.join()

        # Evalulate function here with best network
        # evalulate(network)

        # Perform any additional logic after each iteration
        print("Iteration complete. Proceeding to next iteration.")

if __name__ == "__main__":
    main()