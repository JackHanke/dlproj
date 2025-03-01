from pettingzoo.classic import chess_v6
from utils.agent import Agent
from copy import deepcopy


# records results of num_games games between challenger and current best agent, returns winner of tournament
def evaluator(
    challenger_agent: Agent, 
    current_best_agent: Agent, 
    max_moves: int,
    num_games: int,
    v_resign: float, 
    verbose = False
) -> Agent:

    env = chess_v6.env()  
    player_to_int = {
        "player_0": 1,
        "player_1": -1
    }

    challenger_agent_wins = 0
    current_best_agent_wins = 0

    game_idx = 0
    # TODO implement early stopping
    win_threshold = 0.55
    threshold_games = int(num_games * win_threshold)
    # for game_idx in range(num_games):
    while (game_idx < num_games) or challenger_agent_wins >= threshold_games or current_best_agent_wins >= threshold_games:
        game_idx += 1
        print('*'*50)
        print(f'Starting game #{game_idx}')
        should_disable = False
        supposed_winner = None # For resgin false positive logic
        env.reset()

        # if it is an even game, the challenger plays white
        if (game_idx % 2) == 0:
            player_to_agent = {
                "player_0": challenger_agent,
                "player_1": current_best_agent
            }

        # if it is an even game, the current best plays white
        elif (game_idx % 2) == 1:
            player_to_agent = {
                "player_0": current_best_agent,
                "player_1": challenger_agent
            }

        for move_idx in range(1, max_moves+1):
            print("Move #{}".format(move_idx))
            current_player = env.agent_selection

            observation, reward, termination, truncation, info = env.last()

            # Check for game termination
            if termination:
                print(f"Game terminated for {current_player} at move {move_idx}")
                # env reward is 1 if the current player just won, -1 if just lost, and 0 if drew
                game_result = env.rewards[current_player]  
                # last_player is 1 if current player is white, -1 if current player is black
                last_player = player_to_int[current_player]
                # winning player is 1 if current player is (white and won) or (black and lost), and -1 if (white and lost) or (black and won)
                winning_player = game_result * last_player

                # if challenger wins 
                if ((game_idx % 2) == 0 and winning_player == 1) or ((game_idx % 2) == 1 and winning_player == -1):
                    challenger_agent_wins += 1
                # if best current wins
                elif ((game_idx % 2) == 1 and winning_player == 1) or ((game_idx % 2) == 0 and winning_player == -1):
                    current_best_agent_wins += 1
                else:
                    challenger_agent_wins += 0.5
                    current_best_agent_wins += 0.5

                break

            if truncation:
                print(f"Game truncated for {current_player} at move {move_idx}")
                game_result = 0  
                winning_player = 0
                challenger_agent_wins += 0.5
                current_best_agent_wins += 0.5
                break

            state = observation['observation']
            tau = 1.0 if move_idx < 30 else 0


            # TODO who is player?
            agent = player_to_agent[current_player]
            selected_move, v = agent.inference(board_state=deepcopy(env.board), tau=tau)

            # Resignation logic
            if move_idx > 10:
                if v < v_resign:
                    winning_player = -1 * player_to_int[current_player]
                    break

            # Play the move
            env.step(selected_move)

        else:
            game_result = 0  # Draw by reaching max moves
            winning_player = 0

        if verbose: print(f"Completed game {game_idx + 1}/{num_games}")
        game_idx += 1

    # return best agent
    if challenger_agent_wins > current_best_agent_wins: return challenger_agent
    elif current_best_agent_wins > challenger_agent_wins: return current_best_agent
