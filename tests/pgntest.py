import chess.pgn

game = chess.pgn.Game()

game.headers["Event"] = f"dem0 vs Stockfish"
game.headers["Site"] = f"The CPU"
# v:agent.version
game.headers["White"] = f"Agent 1"
game.headers["Black"] = f"Agent 2"

node = game.add_main_variation(chess.Move.from_uci("e2e4"))
node = node.add_main_variation(chess.Move.from_uci("e7e5"))
node = node.add_main_variation(chess.Move.from_uci("c2c3"))


pgn_str = str(game)
print(pgn_str)

with open("test_game.pgn", "w") as pgn_file:
    pgn_file.write(pgn_str)

with open("test_game.pgn", "r") as pgn_file:
    game = chess.pgn.read_game(pgn_file)
