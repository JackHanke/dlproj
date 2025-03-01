from utils.self_play import SelfPlayAgent
from utils.configs import load_config
from utils.networks import DemoNet
from pettingzoo.classic import chess_v6
from utils.training import train_on_replay_memory
from utils.memory import ReplayMemory, Transition
import torch


def main():
    network = DemoNet(num_res_blocks=1)
    agent = SelfPlayAgent()
    agent.run_self_play(
        network=network,
        n_sims=100,
        num_games=10,
        max_moves=100,
        max_replay_len=10000
    )


def test_env():
    env = chess_v6.env()
    env.reset()
    observation, reward, termination, truncation, info = env.last()
    print("No moves taken yet...")
    print(termination, truncation)
    current_player = env.agent_selection
    print(f"Current Player {current_player}")
    env.step(77)
    print(env.terminations[current_player], env.truncations[current_player])


def test_replay_memory():
    env = chess_v6.env()
    env.reset()
    observation, reward, termination, truncation, info = env.last()
    state = observation['observation']
    memory = ReplayMemory(1000)
    for _ in range(32):
        memory.push(torch.from_numpy(state), torch.randn(4672,), torch.tensor([1]))

    print(len(memory))
    sampled_memory = memory.sample(10)
    batch = Transition(*zip(*sampled_memory))
    state_batch = torch.stack(batch.state)
    policy_batch = torch.stack(batch.policy)
    result_batch = torch.stack(batch.game_result)
    print(state_batch.shape, policy_batch.shape, result_batch.shape)
    print(len(memory))


if __name__ == "__main__":
    test_replay_memory()
