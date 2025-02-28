from collections import deque, namedtuple
import random
from typing import Deque, Literal
import torch

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
    
    def sample_in_batches(self, batch_size: int) -> dict[Literal['state_batch', 'policy_batch', 'reward_batch'], torch.Tensor]:
        """Samples a batch and splits them into 3 items of batches."""
        transition = self.sample(batch_size=batch_size)
        batch = Transition(*zip(*transition))
        return {
            'state_batch': torch.stack(batch.state),
            'policy_batch': torch.stack(batch.policy),
            'reward_batch': torch.stack(batch.game_result)
        }
    
    def __len__(self) -> int:
        """Returns the current size of the memory buffer."""
        return len(self.memory)