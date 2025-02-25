import torch.optim as optim
import torch
import torch.nn as nn
from typing import Literal, Union


def get_optimizer(
    optimizer_name: Literal['adam', 'SGD'],
    lr: float,
    weight_decay: float,
    model: nn.Module,
    momentum: float = 0.0
) -> Union[optim.Adam, optim.SGD]:
    """
    Creates and returns an optimizer for the given model.

    Args:
        optimizer_name (Literal['adam', 'SGD']): 
            The name of the optimizer to use. 
            - 'adam' for Adam optimizer.
            - 'SGD' for Stochastic Gradient Descent (SGD) optimizer.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay (L2 regularization) applied to the optimizer.
        model (nn.Module): The PyTorch model whose parameters will be optimized.
        momentum (float, optional): Momentum factor (only applicable for SGD). Defaults to 0.0.

    Returns:
        Union[optim.Adam, optim.SGD]: The instantiated optimizer.

    Raises:
        ValueError: If an unsupported optimizer name is provided.
    """
    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. Choose either 'adam' or 'SGD'.")