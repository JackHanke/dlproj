import torch
import torch.nn as nn
import torch.nn.functional as F


class DemoNet(nn.Module):
    def __init__(self, input_channels: int = 111, num_res_blocks: int = 5, policy_output_dim: int = 4672):
        """
        AlphaZero-style CNN for PettingZoo Chess Environment
        - input_channels: Number of input feature planes (111)
        - num_res_blocks: Number of residual blocks
        - policy_output_dim: Output size for policy head (4672 for Chess)
        """
        super().__init__()

        # Initial Convolutional Block
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Residual Tower
        self.res_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(num_res_blocks)])

        # Policy Head
        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 2, kernel_size=1),  # Reduce to 2 feature maps
            nn.BatchNorm2d(2),
            nn.ReLU()
        )
        self.policy_fc = nn.Linear(2 * 8 * 8, policy_output_dim)  # Flatten & output move logits

        # Value Head
        self.value_head = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),  # Reduce to 1 feature map
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.value_fc = nn.Sequential(
            nn.Linear(8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for policy and value network.
        :param x: Input tensor of shape (batch_size, 111, 8, 8)
        :return: (policy_logits, value)
        """
        x = self.conv_block(x)  # Initial Conv Layer
        x = self.res_blocks(x)  # Residual Tower

        # Policy Head
        policy = self.policy_head(x)
        policy = policy.view(policy.shape[0], -1)  # Flatten
        policy_logits = self.policy_fc(policy)  # Logits for each move

        # Value Head
        value = self.value_head(x)
        value = value.view(value.shape[0], -1)  # Flatten
        value = self.value_fc(value)

        return policy_logits, value


class ResidualBlock(nn.Module):
    """
    Residual Block as used in AlphaZero.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x  # Skip connection
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # Skip connection
        return self.relu(out)
