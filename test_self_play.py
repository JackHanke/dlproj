from utils.self_play import SelfPlayAgent
from utils.configs import load_config
from utils.networks import DemoNet
from pettingzoo.classic import chess_v6


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


if __name__ == "__main__":
    main()
