
import chess
import chess.engine
from sys import platform

if platform == 'darwin': stock_path = f'./stockfish-5-mac/Mac/stockfish-5-64'
elif platform == 'linux': stock_path = f'./stockfish-5-mac/src/stockfish' 

engine = chess.engine.SimpleEngine.popen_uci(stock_path)
skill_level = 0 # TODO ELO: ~1000-2000
engine.configure({"Skill Level": skill_level})


board = chess.Board()
print(board)

result = engine.play(board, chess.engine.Limit(time=1.0)).move
print(f'Stockfish chose: {result}')
board.push(result)

print(board)

# input()
engine.quit()
