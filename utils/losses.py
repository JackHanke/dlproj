import torch
import torch.nn.functional as F


def policy_loss(pi: torch.Tensor, p_theta_logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(p_theta_logits, dim=-1)
    return (-torch.sum(pi * log_probs, dim=-1)).mean()


def value_loss(z: torch.Tensor, v_theta: torch.Tensor) -> torch.Tensor:
    """
    Computes the value loss using mean squared error (MSE) between the predicted value and the actual game outcome.

    Args:
        z (torch.Tensor): The true outcome of the game (-1 for loss, 0 for draw, 1 for win).
        v_theta (torch.Tensor): The predicted value from the neural network.

    Returns:
        torch.Tensor: Scalar tensor representing the value loss.
    """
    assert v_theta.shape == z.shape
    assert z.ndim == 2
    return F.mse_loss(v_theta, z)  


def combined_loss(
    pi: torch.Tensor, 
    p_theta_logits: torch.Tensor, 
    z: torch.Tensor, 
    v_theta: torch.Tensor,
    policy_weight: float = 1.0, 
    value_weight: float = 1.0, 
) -> torch.Tensor:
    """
    Computes the total loss for AlphaZero training, combining policy loss, and value loss.

    Args:
        pi (torch.Tensor): Target policy from MCTS (probability distribution over actions).
        p_theta_logits (torch.Tensor): Predicted policy from the neural network (raw logits).
        z (torch.Tensor): The true outcome of the game (-1 for loss, 0 for draw, 1 for win).
        v_theta (torch.Tensor): The predicted value from the neural network.
        policy_weight (float): Weighting factor for policy loss (default: 1.0).
        value_weight (float): Weighting factor for value loss (default: 1.0).

    Returns:
        torch.Tensor: Scalar tensor representing the total loss.
    """
    # Compute individual losses
    p_loss = policy_loss(pi, p_theta_logits)
    v_loss = value_loss(z, v_theta)

    # Compute total loss with weights
    total_loss = policy_weight * p_loss + value_weight * v_loss

    return total_loss