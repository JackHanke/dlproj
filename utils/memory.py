import random
import torch
from collections import namedtuple
from typing import List, Dict, Literal

Transition = namedtuple('Transition', ('state', 'policy', 'game_result'))

class ReplayMemory:
    def __init__(self, maxlen: int):
        self.memory = []  # Pure Python list
        self.maxlen = maxlen

    def push(self, *args) -> None:
        """Stores a new transition in memory."""
        if len(self.memory) >= self.maxlen:
            self.memory.pop(0)
        self.memory.append(Transition(*args))

    def load_memory(self, memory_list: List[Transition]) -> None:
        """Loads a list of transitions into memory."""
        for item in memory_list:
            if len(self.memory) >= self.maxlen:
                self.memory.pop(0)
            self.memory.append(item)

    def sample(self, batch_size: int) -> List[Transition]:
        """Samples a batch of transitions from memory."""
        return random.sample(self.memory, batch_size)

    def sample_in_batches(self, batch_size: int) -> Dict[Literal['state_batch', 'policy_batch', 'reward_batch'], torch.Tensor]:
        """Samples a batch and returns as separate tensors."""
        transition = self.sample(batch_size=batch_size)
        if not transition:
            return {
                'state_batch': torch.tensor([]),
                'policy_batch': torch.tensor([]),
                'reward_batch': torch.tensor([])
            }
        batch = Transition(*zip(*transition))
        return {
            'state_batch': torch.stack(batch.state),
            'policy_batch': torch.stack(batch.policy),
            'reward_batch': torch.stack(batch.game_result)
        }

    def __len__(self) -> int:
        """Returns the current size of the memory."""
        return len(self.memory)