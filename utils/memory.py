from collections import namedtuple
import random
import torch
import multiprocessing as mp
from typing import List, Dict, Literal

Transition = namedtuple(
    "Transition",
    ('state', 'policy', 'game_result') 
)

class ReplayMemory:
    def __init__(self, maxlen: int):
        self.manager = mp.Manager()
        self.memory = self.manager.list()  # Shared list between processes
        self.lock = self.manager.Lock()      # Shared lock via manager
        self.maxlen = maxlen

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the manager before pickling since it is not picklable.
        state.pop("manager", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def push(self, *args) -> None:
        """Stores a new transition in memory."""
        with self.lock:
            if len(self.memory) >= self.maxlen:
                self.memory.pop(0)  # Maintain fixed size
            self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        """Samples a batch of transitions from memory safely."""
        with self.lock:
            return random.sample(list(self.memory), batch_size)

    def sample_in_batches(self, batch_size: int) -> Dict[Literal['state_batch', 'policy_batch', 'reward_batch'], torch.Tensor]:
        """Samples a batch and splits them into separate tensors."""
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
        """Returns the current size of the memory buffer."""
        with self.lock:
            return len(self.memory)