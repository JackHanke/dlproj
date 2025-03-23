import argparse
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
from utils.self_play import SelfPlaySession
from utils.memory import ReplayMemory
from utils.networks import DemoNet
from utils.optimizer import get_optimizer  
from utils.evaluator import evaluator
from utils.agent import Agent, Stockfish
from utils.utils import Timer
from utils.configs import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run AlphaZero-like training loop.")
    parser.add_argument("--mp_training", action="store_true", help="Use multiprocessing for training.")
    parser.add_argument("--start_with_empty_replay_memory", action="store_true", help="Start with empty replay memory.")
    parser.add_argument("--no_self_play", dest="run_self_play", action="store_false", help="Skip self play.")
    parser.add_argument("--no_train", dest="train_network", action="store_false", help="Skip training.")
    parser.add_argument("--train_iterations", type=int, default=50, help="Number of training iterations if not using multiprocessing.")
    return parser.parse_args()


def training_loop(stop_event, memory, network, device, optimizer_params, counter, batch_size, iteration, checkpoint: Checkpoint):
    logging.basicConfig(filename='dem0.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    optimizer = get_optimizer(
        optimizer_name=optimizer_params["optimizer_name"],
        lr=optimizer_params["lr"],
        weight_decay=optimizer_params["weight_decay"],
        model=network,
        momentum=optimizer_params.get("momentum", 0.0)
    )
    i = 0
    while not stop_event.is_set():
        r = train_on_batch(
            data=memory,
            network=network,
            batch_size=batch_size,
            device=device,
            optimizer=optimizer
        )
        if r:
            i += 1
            if i == 1:
                logging.info('Training started!')
            if i % 10 == 0:
                checkpoint.save_state_dict(iteration=iteration, state_dict=network.state_dict())
    counter.value = i
    logging.info(f"Trained on {i} batches.")
    logging.info(f'Memory length after self play: {len(memory)}')


def get_latest_iteration(checkpoint_client: Checkpoint):
    folders: list[str] = checkpoint_client.list_folder_names()
    if not folders:
        return 0
    versions = map(lambda x: int(x.split('/')[-1].removeprefix("iteration_")), folders)
    return max(versions)


def main(args):
    os.system("./clear_log.sh")
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='dem0.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f'\n\nRunning Experiment on {datetime.now()} with the following configs:')
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("azure.storage").setLevel(logging.WARNING)
    checkpoint = Checkpoint(verbose=True, compute_elo=False)

    latest_iteration = get_latest_iteration(checkpoint_client=checkpoint)
    if latest_iteration == 0:
        logging.info("Starting from iterations from scratch.")
        stockfish_level = 0
        stockfish_progress = {0:[]}
        current_best_version = 0
        self_play_start_from = 0
    else:
        stockfish_level = None
        current_best_version = 0
        try:
            self_play_start_from = checkpoint.download_from_blob(blob_name=f"checkpoints/iteration_{latest_iteration}/self_play_games_completed.pkl")
            logger.info(f"Starting self play from game {self_play_start_from}.")
        except FileNotFoundError:
            logger.warning("Starting self play from game 0.")
            self_play_start_from = 0
        _iter = latest_iteration
        while _iter > 0:
            try:
                info = checkpoint.download_from_blob(f"checkpoints/iteration_{_iter}/info.json")
                logger.info(f'Found info.json for latest version {latest_iteration}.')
                stockfish_level = info['stockfish_level']
                logger.info(f'Found stockfish level {stockfish_level} in info.json for latest iteration {latest_iteration}.')
                break
            except Exception as e:
                logger.warning(f'Failed getting info from iteration {_iter}: {e}')
                if stockfish_level is None:
                    stockfish_level = 0
                _iter -= 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = load_config()
    logger.info("Device for network training:", device)

    self_play_session = SelfPlaySession(checkpoint_client=checkpoint)
    memory = ReplayMemory(configs.training.data_buffer_size)
    if not args.start_with_empty_replay_memory:
        if checkpoint.blob_exists(f'checkpoints/iteration_{latest_iteration}/replay_memory.pkl'):
            memory_list = checkpoint.load_replay_memory(iteration=latest_iteration)
            memory.load_memory(memory_list)
            logger.info(f"Loaded memory from blob path 'checkpoints/iteration_{latest_iteration}/replay_memory.pkl' with length = {len(memory)}")
        else:
            logger.info("Starting with empty memory.")
    else:
        logger.info('Starting with empty memory.')

    optimizer_params = {
        "optimizer_name": configs.training.optimizer,
        "lr": configs.training.learning_rate.initial,
        "weight_decay": configs.training.weight_decay,
        "momentum": configs.training.momentum
    }

    current_best_network = DemoNet(num_res_blocks=configs.network.num_residual_blocks)
    challenger_network = deepcopy(current_best_network)
    challenger_network.share_memory()

    if checkpoint.blob_exists(f'{checkpoint.best_path}/weights.pth'):
        pretrained_weights = checkpoint.download_from_blob(f'{checkpoint.best_path}/weights.pth', device=device)
        logger.info(f"Loaded weights from blob path: {checkpoint.best_path}/weights.pth")
        current_best_version = checkpoint.download_from_blob(f'{checkpoint.best_path}/info.json', device=device)['version']
    elif checkpoint.blob_exists('checkpoints/pretrained_weights.pth'):
        pretrained_weights = checkpoint.download_from_blob('checkpoints/pretrained_weights.pth', device=device)
        logger.info(f"Loaded weights from blob path: checkpoints/pretrained_weights.pth")
    else:
        pretrained_weights = None
        assert pretrained_weights

    i = latest_iteration
    while True:
        iter_start_time = time.time()
        logger.info(f'Beginning dem0 Iteration {i+1}...\n')

        current_best_agent = Agent(version=current_best_version, network=current_best_network, sims=configs.self_play.num_simulations)
        challenger_agent = Agent(version=current_best_version+1, network=challenger_network, sims=configs.self_play.num_simulations)

        if args.mp_training:
            stop_event = mp.Event()
            counter = mp.Value("i", 0)
            training_process = mp.Process(
                target=training_loop,
                args=(stop_event, memory, challenger_network, device, optimizer_params, counter, configs.training.batch_size, i+1, checkpoint)
            )
            training_process.start()

            self_play_session.run_self_play(
                iteration=i+1,
                training_data=memory,
                network=current_best_network,
                device=device,
                n_sims=configs.self_play.num_simulations,
                num_games=configs.training.num_self_play_games,
                max_moves=300,
                start_from_game_idx=self_play_start_from
            )

            self_play_start_from = 0
            stop_event.set()
            training_process.join()
            logger.debug(f'Training iterations completed: {counter.value}')

        else:
            if args.run_self_play:
                logger.info("Starting basic self play...")
                self_play_session.run_self_play(
                    training_data=memory,
                    network=current_best_network,
                    device=device,
                    n_sims=configs.self_play.num_simulations,
                    num_games=configs.training.num_self_play_games,
                    max_moves=300
                )
            if args.train_network:
                for b in tqdm(range(args.train_iterations)):
                    _ = train_on_batch(
                        data=memory,
                        network=challenger_network,
                        batch_size=configs.training.batch_size,
                        optimizer=get_optimizer(model=challenger_network, **optimizer_params),
                        device=device
                    )
                    if (b+1) % 5 == 0:
                        checkpoint.save_state_dict(iteration=i+1, state_dict=challenger_network.state_dict())
                logger.debug(f'Training iterations completed: {args.train_iterations}')

        # === Evaluation ===
        weights_changed = any(
            not torch.equal(p1, p2) for p1, p2 in zip(challenger_agent.network.parameters(), current_best_agent.network.to(device).parameters())
        )
        assert weights_changed, "Error: Challenger network weights did not change after training!"
        logger.info("✅ Challenger network weights updated successfully.")

        new_best_agent, wins, draws, losses, win_percent, tot_games = evaluator(
            challenger_agent=challenger_agent,
            current_best_agent=current_best_agent,
            device=device,
            max_moves=300,
            num_games=configs.evaluation.tournament_games,
            v_resign=self_play_session.v_resign,
            win_threshold=configs.evaluation.evaluation_threshold
        )
        logger.info(f'Agent {challenger_agent.version} playing Agent {current_best_agent.version}, won {wins} games, drew {draws} games, lost {losses} games. ({round(100*win_percent, 2)}% wins.)')

        current_best_network = deepcopy(new_best_agent.network).to(device)
        challenger_network = new_best_agent.network.to(device)
        current_best_version = new_best_agent.version

        stockfish = Stockfish(level=stockfish_level)
        wins, draws, losses, win_percent, tot_games = evaluator(
            challenger_agent=new_best_agent,
            current_best_agent=stockfish,
            device=device,
            max_moves=300,
            num_games=configs.evaluation.tournament_games,
            v_resign=self_play_session.v_resign
        )
        logger.info(f'Against Stockfish 5 Level {stockfish_level}, won {wins} games, drew {draws} games, lost {losses} games. ({round(100*win_percent, 2)}% wins.)')

        stockfish_progress[stockfish_level].append(win_percent)
        if win_percent > 0.55:
            stockfish_level += 1
            stockfish.engine.configure({"Skill Level": stockfish_level})
            logger.info(f'! Upgrading Stockfish level to {stockfish_level}.')

        checkpoint.step(
            current_best_agent=deepcopy(new_best_agent),
            memory=memory,
            info={
                "iteration": i+1,
                "version": new_best_agent.version,
                "stockfish_level": stockfish_level,
                "stockfish_eval": f'Against Stockfish 5 Level {stockfish_level}, won {wins} games, drew {draws} games, lost {losses} games. ({round(100*win_percent, 2)}% wins.)',
                "self_eval": f'Agent {challenger_agent.version} played Agent {current_best_agent.version}, won {wins} games, drew {draws} games, lost {losses} games. ({round(100*win_percent, 2)}% wins.)'
            },
            current_iteration=i+1
        )

        i += 1
        stockfish.engine.close()


def run():
    args = parse_args()
    with Timer():
        mp.set_start_method("spawn")
        main(args)



if __name__ == "__main__":
    run()