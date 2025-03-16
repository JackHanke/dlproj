import numpy as np
import chess
from pettingzoo.classic import chess_v6
from utils.chess_utils_local import get_action_mask, get_observation, legal_moves as legal_moves_func
from utils.utils import observe


def compare_observations():
    board_history = np.zeros((8, 8, 104), dtype=bool)
    # Initialize PettingZoo Chess environment
    env = chess_v6.env()
    env.reset()

    i = 0
    correct = 0
    while True:
        i += 1

        # Get PettingZoo's observation
        obs, reward, termination, truncation, info = env.last()
        print(f'Termination: {termination}, Truncation: {truncation}, Reward: {reward}')
        pettingzoo_obs_array = obs["observation"]
        pettingzoo_action_mask = obs["action_mask"]

        # Get the board state from PettingZoo
        board = env.board

        # Generate observation using your function
        obs = observe(board=board, agent=env.agent_selection, agent_selection=env.agent_selection ,possible_agents=env.possible_agents, board_history=board_history)
        my_obs_array = obs['observation']
        my_action_mask = obs['action_mask']

        # Compare observations
        obs_equal = np.array_equal(my_obs_array, pettingzoo_obs_array)
        action_mask_equal = np.array_equal(my_action_mask, pettingzoo_action_mask)

        if obs_equal and action_mask_equal:
            print("✅ Observations and action masks match!")
            correct += 1
        else:
            print("❌ Observations or action masks do not match!")
            
        print((my_obs_array == pettingzoo_obs_array).mean())

        if truncation or termination:
             print(f"We got {i}/{correct} correct.")
             break
        actions = np.where(obs['action_mask'])[0]
        action = np.random.choice(actions)
        env.step(action)
        next_board = get_observation(board, player=0)
        board_history = np.dstack(
            (next_board[:, :, 7:], board_history[:, :, :-13])
        )

# Run the comparison
if __name__ == '__main__':
    compare_observations()