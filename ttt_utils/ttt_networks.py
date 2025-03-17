import torch
import torch.nn as nn
import torch.nn.functional as F


class DemoTicTacToeConvNet(nn.Module):
    def __init__(self, input_channels: int = 2, num_res_blocks: int = 1, policy_output_dim: int = 9):
        """
        AlphaZero-style CNN for PettingZoo Chess Environment
        - input_channels: Number of input feature planes (111)
        - num_res_blocks: Number of residual blocks
        - policy_output_dim: Output size for policy head (4672 for Chess)
        """
        super().__init__()
        self.net_width = 16

        # Initial Convolutional Block
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels, self.net_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.net_width),
            nn.ReLU()
        )

        # Residual Tower
        self.res_blocks = nn.Sequential(*[ResidualBlock(self.net_width) for _ in range(num_res_blocks)])

        # Policy Head
        self.policy_head = nn.Sequential(
            nn.Conv2d(self.net_width, 2, kernel_size=1),  # Reduce to 2 feature maps
            nn.BatchNorm2d(2),
            nn.ReLU()
        )
        self.policy_fc = nn.Linear(18, policy_output_dim)  # Flatten & output move logits

        # Value Head
        self.value_head = nn.Sequential(
            nn.Conv2d(self.net_width, 1, kernel_size=1),  # Reduce to 1 feature map
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.value_fc = nn.Sequential(
            nn.Linear(9, self.net_width),
            nn.ReLU(),
            nn.Linear(self.net_width, 1),
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

class DemoTicTacToeFeedForwardNet(nn.Module):
    def __init__(self, input_size: int = 18, num_layers: int = 1, policy_output_dim: int = 9):
        """
        AlphaZero-style CNN for PettingZoo Chess Environment
        - input_channels: Number of input feature planes (111)
        - num_res_blocks: Number of residual blocks
        - policy_output_dim: Output size for policy head (4672 for Chess)
        """
        super().__init__()
        self.net_width = 16

        self.first = nn.Linear(input_size, self.net_width)

        # Residual Tower
        self.res_blocks = nn.Sequential(*[nn.Linear(self.net_width, self.net_width) for _ in range(num_layers)])

        # Policy Head
        self.policy_head = nn.Sequential(
            nn.Linear(self.net_width, self.net_width),
            nn.ReLU(),
            nn.Linear(self.net_width, policy_output_dim),
            nn.ReLU()
        )

        # Value Head
        self.value_head = nn.Sequential(
            nn.Linear(self.net_width, self.net_width),
            nn.ReLU(),
            nn.Linear(self.net_width, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for policy and value network.
        :param x: Input tensor of shape (batch_size, 111, 8, 8)
        :return: (policy_logits, value)
        """
        x = torch.reshape(x, (-1, 18))
        x = self.first(x)
        x = self.res_blocks(x)  # Residual Tower

        # Policy Head
        policy_logits = self.policy_head(x)  # Logits for each move

        # Value Head
        value = self.value_head(x)

        return policy_logits, value


if __name__ == '__main__':
    num_res_blocks = 2
    conv_model = DemoTicTacToeConvNet(num_res_blocks=num_res_blocks)
    
    num_layers = 2
    ffnn_model = DemoTicTacToeFeedForwardNet(num_layers=num_layers)

    total_params = sum(p.numel() for p in conv_model.parameters())
    print(f"Number of parameters for conv num_res_blocks = {num_res_blocks}: {total_params}")
    total_params_ffnn = sum(p.numel() for p in ffnn_model.parameters())
    print(f"Number of parameters for num_layers = {num_layers}: {total_params_ffnn}")

