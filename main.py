import argparse
import torch
import torch.nn.utils as nn_utils
import torch.multiprocessing as mp
import time
import os
from tqdm import tqdm
from copy import deepcopy
import logging
from datetime import datetime

from utils.training import train_on_batch, Checkpoint
from utils.self_play import SelfPlaySession
from utils.memory import ReplayMemory
from utils.networks import DemoNet
from utils.optimizer import get_optimizer  
from utils.evaluator import evaluator
from utils.agent import Agent, Stockfish
from utils.utils import Timer
from utils.configs import load_config
from typing import Literal


def parse_args():
    parser = argparse.ArgumentParser(description="Run AlphaZero-like training loop.")
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument("--mp_training", action="store_true", help="Use multiprocessing for training.")
    parser.add_argument("--start_with_empty_replay_memory", action="store_true", help="Start with empty replay memory.")
    return parser.parse_args()


def training_loop(stop_event, transition_queue, network, device, optimizer_params, batch_size, counter, checkpoint, iteration, max_replay_memory_size):
    logging.basicConfig(filename='dem0.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    optimizer = get_optimizer(
        optimizer_name=optimizer_params["optimizer_name"],
        lr=optimizer_params["lr"],
        weight_decay=optimizer_params.get("weight_decay", 0.0),
        model=network,
        momentum=optimizer_params.get("momentum", 0.0)
    )

    memory = ReplayMemory(max_replay_memory_size)

    i = 0
    while not stop_event.is_set() or not transition_queue.empty():
        while not transition_queue.empty():
            s, pi, r = transition_queue.get()
            memory.push(s, pi, r)

        trained, loss, replay_size = train_on_batch(
            data=memory,
            network=network,
            batch_size=batch_size,
            device=device,
            optimizer=optimizer
        )

        if trained:
            i += 1
            if i == 1:
                logging.info("Training started!")
            if i % 10 == 0:
                logging.info(f"Batch {i}: loss = {loss.item():.4f}, total replay size: {replay_size}")
                checkpoint.save_state_dict(iteration=iteration, state_dict=network.state_dict())
                checkpoint.save_replay_memory(memory=memory, iteration=iteration)

    # 🔥 Save memory after training ends
    checkpoint.save_replay_memory(memory=memory, iteration=iteration)
    logging.info(f"✅ Saved ReplayMemory after training for iteration {iteration}.")

    counter.value = i
    logging.info(f"Training finished after {i} batches.")
    logging.info(f"Final memory length = {len(memory)}")


def get_latest_iterations(checkpoint_client: Checkpoint) -> dict[Literal['latest_completed_checkpoint', 'latest_started_checkpoint'], int]:
    folders: list[str] = checkpoint_client.list_folder_names()
    if not folders:
        return {"latest_completed_checkpoint": 0, "latest_started_checkpoint": 0}

    versions = map(lambda x: int(x.split('/')[-1].removeprefix("iteration_")), folders)
    latest_started_checkpoint = max(versions)
    if checkpoint_client.blob_exists(f"checkpoints/iteration_{latest_started_checkpoint}/info.json"):
        latest_completed_checkpoint = latest_started_checkpoint
    else:
        latest_completed_checkpoint = latest_started_checkpoint - 1

    return {
        "latest_completed_checkpoint": latest_completed_checkpoint,
        "latest_started_checkpoint": latest_started_checkpoint
    }


def main(args):
    os.system("./clear_log.sh")
    logging.basicConfig(filename='dem0.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(f'\n\nRunning Experiment on {datetime.now()} with the following configs:')
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("azure.storage").setLevel(logging.WARNING)

    checkpoint = Checkpoint(verbose=False)
    configs = load_config()
    global MAX_REPLAY_MEMORY_SIZE
    MAX_REPLAY_MEMORY_SIZE = configs.training.data_buffer_size

    iteration_dict = get_latest_iterations(checkpoint_client=checkpoint)

    # Which iteration to start
    if iteration_dict['latest_started_checkpoint'] == 0:
        i = 1
        self_play_start_from = 0
        stockfish_level = 0
    elif iteration_dict['latest_started_checkpoint'] == 1 and iteration_dict['latest_completed_checkpoint'] == 0:
        i = 1
        self_play_start_from = checkpoint.download_from_blob(
            blob_name=f"checkpoints/iteration_{iteration_dict['latest_started_checkpoint']}/self_play_games_completed.pkl"
        )
        stockfish_level = 0
    elif iteration_dict["latest_completed_checkpoint"] == iteration_dict["latest_started_checkpoint"]:
        stockfish_level = checkpoint.download_from_blob(
            blob_name=f"checkpoints/iteration_{iteration_dict['latest_completed_checkpoint']}/info.json"
        )['stockfish_level']
        i = iteration_dict["latest_completed_checkpoint"] + 1
        self_play_start_from = 0
    else:
        stockfish_level = checkpoint.download_from_blob(
            blob_name=f"checkpoints/iteration_{iteration_dict['latest_completed_checkpoint']}/info.json"
        )['stockfish_level']
        i = iteration_dict['latest_started_checkpoint']
        self_play_start_from = checkpoint.download_from_blob(
            blob_name=f"checkpoints/iteration_{iteration_dict['latest_started_checkpoint']}/self_play_games_completed.pkl"
        )

    logger.info(f"Resuming at iteration {i}, starting from self-play game {self_play_start_from + 1}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    transition_queue = mp.Queue()

    # === If memory exists, prefill the queue ===
    if not args.start_with_empty_replay_memory:
        if checkpoint.blob_exists(f"checkpoints/iteration_{iteration_dict['latest_started_checkpoint']}/replay_memory.pkl"):
            memory_list = checkpoint.load_replay_memory(iteration=iteration_dict['latest_started_checkpoint'])
            for item in memory_list:
                transition_queue.put(item)
            logger.info(f"Loaded {len(memory_list)} items into transition queue.")
        else:
            logger.info("No previous memory found, starting empty.")
    else:
        logger.info("Starting from empty memory as requested.")

    logger.info(f'Starting at iteration {i}.')

    current_best_network = DemoNet(num_res_blocks=configs.network.num_residual_blocks)
    challenger_network = DemoNet(num_res_blocks=configs.network.num_residual_blocks)

    if checkpoint.blob_exists(f'{checkpoint.best_path}/weights.pth'):
        pretrained_weights = checkpoint.download_from_blob(f'{checkpoint.best_path}/weights.pth', device=device)
        current_best_network.load_state_dict(pretrained_weights)
        current_best_version = checkpoint.download_from_blob(f'{checkpoint.best_path}/info.json', device=device)['version']
        logger.info(f"Loaded best model version {current_best_version}.")
    else:
        current_best_version = 0

    checkpoint.load_challenger_weights_if_available(i, challenger_network, fallback_network=current_best_network, logger=logger)

    self_play_session = SelfPlaySession(checkpoint_client=checkpoint)

    optimizer_params = {
        "optimizer_name": configs.training.optimizer,
        "lr": configs.training.learning_rate.initial,
        "weight_decay": configs.training.weight_decay,
        "momentum": configs.training.momentum
    }

    while True:
        logger.info(f"=== Starting iteration {i} ===")

        # --- reset and refill queue ---
        if transition_queue.empty():
            if checkpoint.blob_exists(f"checkpoints/iteration_{i}/replay_memory.pkl"):
                memory_list = checkpoint.load_replay_memory(iteration=i)
                for state, policy, reward in memory_list:
                    transition_queue.put((state, policy, reward))
                logger.info(f"Loaded {len(memory_list)} items into transition queue for iteration {i}.")
            elif checkpoint.blob_exists(f"checkpoints/iteration_{i-1}/replay_memory.pkl"):
                memory_list = checkpoint.load_replay_memory(iteration=i-1)
                for state, policy, reward in memory_list:
                    transition_queue.put((state, policy, reward))
                logger.info(f"Loaded {len(memory_list)} items into transition queue for iteration {i} from iteration {i-1}.")
            else:
                logger.info(f"No saved ReplayMemory found for iteration {i}, starting empty.")

        current_best_agent = Agent(version=current_best_version, network=current_best_network, sims=configs.self_play.num_simulations)
        challenger_agent = Agent(version=current_best_version + 1, network=challenger_network, sims=configs.self_play.num_simulations)

        stop_event = mp.Event()
        counter = mp.Value("i", 0)
        training_process = mp.Process(
            target=training_loop,
            args=(stop_event,
                  transition_queue,
                  challenger_network,
                  device,
                  optimizer_params,
                  configs.training.batch_size,
                  counter,
                  checkpoint,
                  i,
                  MAX_REPLAY_MEMORY_SIZE)
        )
        training_process.start()

        self_play_session.run_self_play(
            iteration=i,
            transition_queue=transition_queue,
            network=current_best_network,
            device=device,
            n_sims=configs.self_play.num_simulations,
            num_games=configs.training.num_self_play_games,
            max_moves=configs.training.max_moves,
            start_from_game_idx=self_play_start_from
        )

        self_play_start_from = 0

        stop_event.set()
        training_process.join()
        logger.info(f"Training batches completed: {counter.value}")

        # === Evaluation ===
        weights = checkpoint.download_from_blob(f"checkpoints/iteration_{i}/weights.pth", device=device)
        challenger_agent.network.load_state_dict(weights)

        # 🚨 Assert that challenger weights changed from current best
        current_flat = nn_utils.parameters_to_vector(current_best_network.parameters())
        challenger_flat = nn_utils.parameters_to_vector(challenger_agent.network.parameters())
        diff_norm = (current_flat - challenger_flat).norm().item()
        logger.info(f"Challenger vs Current Best weight diff norm = {diff_norm:.6f}")
        assert diff_norm > 1e-6, f"Challenger network weights did not change after training! (diff_norm={diff_norm:.6f})"

        new_best_agent, wins, draws, losses, win_percent, _ = evaluator(
            challenger_agent=challenger_agent,
            current_best_agent=current_best_agent,
            device=device,
            max_moves=configs.training.max_moves,
            num_games=configs.evaluation.tournament_games,
            checkpoint_client=checkpoint,
            iteration=i,
            win_threshold=configs.evaluation.evaluation_threshold
        )

        self_eval = f'Agent {challenger_agent.version} vs {current_best_agent.version}: {wins} wins, {draws} draws, {losses} losses ({round(100 * win_percent, 2)}% winrate).'
        logger.info(self_eval)

        current_best_network = deepcopy(new_best_agent.network).to(device)
        challenger_network = deepcopy(new_best_agent.network).to(device)
        current_best_version = new_best_agent.version

        # === Stockfish eval ===
        stockfish = Stockfish(level=stockfish_level)
        s_wins, s_draws, s_losses, s_win_percent, _ = evaluator(
            challenger_agent=new_best_agent,
            current_best_agent=stockfish,
            device=device,
            max_moves=configs.training.max_moves,
            num_games=configs.evaluation.tournament_games,
            checkpoint_client=checkpoint,
            iteration=i
        )

        stockfish_eval = f'Against Stockfish level {stockfish_level}: {s_wins} wins, {s_draws} draws, {s_losses} losses ({round(100 * s_win_percent, 2)}% winrate).'
        logger.info(stockfish_eval)

        if win_percent > 0.55:
            stockfish_level += 1
            logger.info(f'Upgraded Stockfish level to {stockfish_level}.')

        checkpoint.step(
            current_best_agent=deepcopy(new_best_agent),
            info={
                "iteration": i,
                "version": new_best_agent.version,
                "stockfish_level": stockfish_level,
                "stockfish_eval": stockfish_eval,
                "self_eval": self_eval
            },
            current_iteration=i
        )

        i += 1
        stockfish.engine.close()


def run():
    args = parse_args()
    with Timer():
        mp.set_start_method("spawn", force=True)
        main(args)


if __name__ == "__main__":
    run()