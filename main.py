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


def training_loop(stop_event, memory, network, device, optimizer_params, counter, batch_size):
    # Ensure logging is configured in child processes
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
                logging.info('Training started!')  # Should appear in log now

    counter.value = i  
    logging.info(f"Trained on {i} batches.")
    logging.info(f'Memory length after self play: {len(memory)}')

# mp_training is whether or not to use multiprocessing training to run while self-play runs
def main(mp_training: bool = True, start_with_empty_replay_memory: bool = False, run_self_play: bool = True, train_network: bool = True):
    os.system("./clear_log.sh")
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='dem0.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f'\n\nRunning Experiment on {datetime.now()} with the following configs:')
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("azure.storage").setLevel(logging.WARNING)
    checkpoint = Checkpoint(verbose=True, compute_elo=False)
    # TODO add all configs to logging for this log

    base_path = "checkpoints/best_model/"
    weights_path = os.path.join(base_path, "weights.pth")
    info_path = os.path.join(base_path, "info.json")
    model_path = os.path.join(base_path, "model.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = load_config()
    print("Device for network training:", device)

    self_play_session = SelfPlaySession(checkpoint_client=checkpoint)
    memory = ReplayMemory(configs.training.data_buffer_size)
    if not start_with_empty_replay_memory:
        if checkpoint.blob_exists('checkpoints/replay_memory.pkl'):
            memory_list = checkpoint.load_replay_memory()
            memory.load_memory(memory_list)
            print(f"Loaded memory from blob path checkpoints/replay_memory.pkl with length = {len(memory)}")
        else:
            print("Starting with empty memory.")
    else:
        print('Starting with empty memory.')
    optimizer_params = {
        "optimizer_name": configs.training.optimizer,
        "lr": configs.training.learning_rate.initial,
        "weight_decay": configs.training.weight_decay,
        "momentum": configs.training.momentum  
    }
        
    current_best_network = DemoNet(num_res_blocks=configs.network.num_residual_blocks)

    challenger_network = deepcopy(current_best_network)
    challenger_network.share_memory()
    # checkpoint.save_pretrained(state_dict=pretrained_weights)
    if checkpoint.blob_exists('checkpoints/weights.pth'):
        pretrained_weights = checkpoint.download_from_blob('checkpoints/weights.pth', return_bytes=False, device=device)
        print(f"Loaded weights from blob path: checkpoints/weights.pth")
    elif checkpoint.blob_exists('checkpoints/pretrained_weights.pth'):
        pretrained_weights = checkpoint.download_from_blob('checkpoints/pretrained_weights.pth', return_bytes=False, device=device)
        print(f"Loaded weights from blob path: checkpoints/pretrained_weights.pth")
    else:
        pretrained_weights = None
        assert pretrained_weights

    stockfish_level = 0
    stockfish_progress = {0:[]}
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
            sims=configs.self_play.num_simulations
        )
        challenger_agent = Agent(
            version=current_best_version+1, 
            network=challenger_network, 
            sims=configs.self_play.num_simulations
        )
        
        # NOTE multiprocessing code
        if mp_training:
            stop_event = mp.Event()
            counter = mp.Value("i", 0)  # Shared integer for storing i
            training_process = mp.Process(
                target=training_loop,
                args=(stop_event, memory, challenger_network, device, optimizer_params, counter, configs.training.batch_size)
            )
            training_process.start()

            # 
            self_play_session.run_self_play(
                training_data=memory,
                network=current_best_network,
                device=device,
                n_sims=configs.self_play.num_simulations,
                num_games=configs.training.num_self_play_games,
                max_moves=300
            )
            
            stop_event.set()
            training_process.join()
            # print(f"\nTraining iterations completed: {counter.value}")  # Print the value of i
            logger.debug(f'Training iterations completed: {counter.value}')
        else:
            if run_self_play:
                 print("Starting basic self play...")
                 self_play_session.run_self_play(
                    training_data=memory,
                    network=current_best_network,
                    device=device,
                    n_sims=configs.self_play.num_simulations,
                    num_games=configs.training.num_self_play_games,
                    max_moves=300
                )
            if train_network:
                for b in tqdm(range(50)):
                    _ = train_on_batch(
                        data=memory,
                        network=challenger_network,
                        batch_size=configs.training.batch_size,
                        optimizer=get_optimizer(model=challenger_network, **optimizer_params),
                        device=device
                    )
                    if (b+1) % 5 == 0:
                        checkpoint.save_state_dict(
                            path=f"version_{challenger_agent.version}/model_weights.pth",
                            state_dict=challenger_network.state_dict()
                        )
                logger.debug(f'Training iterations completed: {b}')


        # Assert that weights have changed
        if mp_training or train_network:
            weights_changed = any(
                not torch.equal(p1, p2) for p1, p2 in zip(challenger_agent.network.parameters(), current_best_agent.network.to(device).parameters())
            )

            assert weights_changed, "Error: Challenger network weights did not change after training!"
            print("✅ Challenger network weights updated successfully.")

        # print("\nEvaluating...")
        new_best_agent, wins, draws, losses, win_percent, tot_games = evaluator(
            challenger_agent=challenger_agent, 
            current_best_agent=current_best_agent,
            device=device,
            max_moves=300,
            num_games=configs.evaluation.tournament_games,
            v_resign=self_play_session.v_resign,
            win_threshold=configs.evaluation.evaluation_threshold
        )
        # print(f'After this loop, the best_agent is {current_best_agent.version}\n\n')
        logger.info(f'Agent {challenger_agent.version} playing Agent {current_best_agent.version}, won {wins} games, drew {draws} games, lost {losses} games. ({round(100*win_percent, 2)}% wins.)')

        current_best_network = deepcopy(new_best_agent.network).to(device)
        challenger_network = new_best_agent.network.to(device)
        current_best_version = new_best_agent.version

        # print("\nExternal evaluating...")
        stockfish = Stockfish(level=stockfish_level)
        wins, draws, losses, win_percent, tot_games = evaluator(
            challenger_agent=new_best_agent,
            current_best_agent=stockfish,
            device=device,
            max_moves=300,
            num_games=configs.evaluation.tournament_games,
            v_resign=self_play_session.v_resign
        )
        # print(f'Against Stockfish 5 Level {stockfish_level}, won {win_percent} games, lost {loss_percent} games.')
        logger.info(f'Against Stockfish 5 Level {stockfish_level}, won {wins} games, drew {draws} games, lost {losses} games. ({round(100*win_percent, 2)}% wins.)')
        
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
            # plt.plot([i+offset for i in range(len(progress))], progress, label=f'Stockfish Level {level}')
            offset += len(progress)

            # plt.ion()
            # self_play_games = 10 # TODO change this to config var
            # plt.xlabel(f'Training Loops ({self_play_games} games per)')
            # plt.ylabel(f'Win Percentage')
            # plt.title('dem0 Training Against Stockfish Levels')
            # plt.show(block=False)

        # step checkpoint
        checkpoint.step(
            current_best_agent=deepcopy(new_best_agent), 
            memory=memory,
            info={
                "stockfish_eval": f'Against Stockfish 5 Level {stockfish_level}, won {wins} games, drew {draws} games, lost {losses} games. ({round(100*win_percent, 2)}% wins.)',
                "self_eval": f'Agent {challenger_agent.version} played Agent {current_best_agent.version}, won {wins} games, drew {draws} games, lost {losses} games. ({round(100*win_percent, 2)}% wins.)'
            }
        )

        i += 1
    
    stockfish.engine.close()

if __name__ == "__main__":
    with Timer():
        mp.set_start_method("spawn")  
        main(
            mp_training=False,
            start_with_empty_replay_memory=True,
            run_self_play=True,
            train_network=False
        )
