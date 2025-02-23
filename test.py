from pettingzoo.classic import chess_v6

env = chess_v6.env(render_mode="human")
env.reset(seed=42)

observation, reward, termination, truncation, info = env.last()

print(observation['observation'].shape)