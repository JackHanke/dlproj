
# NOTE
def evaluator(challenger_agent, current_best_agent):

    # TODO implement early stopping

    win_threshold = 0.55
    challenger_prob = 0 # TODO 
    if challenger_prob >= win_threshold: 
        best_agent = challenger_agent
        del challenger_agent # TODO kill competitor, is this how you do it?
    elif challenger_prob < win_threshold: 
        best_agent = current_best_agent
        del challenger_agent

    return best_agent
