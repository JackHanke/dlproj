from collections import deque, namedtuple
import random
from typing import Deque

Transition = namedtuple(
    "Transition",
    ('state', 'policy', 'game_result') 
)


class ReplayMemory:
    def __init__(self, maxlen: int):
        self.memory: Deque[Transition] = deque([], maxlen=maxlen)

    def push(self, *args) -> None:
        """Stores a new transition in memory."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> list[Transition]:
        """Samples a batch of transitions from memory."""
        return random.sample(self.memory, batch_size)
    
    def __len__(self) -> int:
        """Returns the current size of the memory buffer."""
        return len(self.memory)