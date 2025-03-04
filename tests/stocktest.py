
import chess
import chess.engine


# 
stock_path = f'/Users/jack/vault/gradschool/dl/dlproj/stockfish-5-mac/Mac/stockfish-5-64'
engine = chess.engine.SimpleEngine.popen_uci(stock_path)
skill_level = 1 # TODO ELO: ~1000-2000
engine.configure({"Skill Level": skill_level})


board = chess.Board()
print(board)

result = engine.play(board, chess.engine.Limit(time=1.0)).move
print(f'Stockfish chose: {result}')
board.push(result)

print(board)

input()
engine.quit()
