import yaml
from dataclasses import dataclass


@dataclass
class LearningRateConfig:
    initial: float


@dataclass
class LossWeightsConfig:
    policy_loss: float
    value_loss: float


@dataclass
class TrainingConfig:
    learning_rate: LearningRateConfig
    batch_size: int
    momentum: float
    weight_decay: float
    loss_weights: LossWeightsConfig
    training_steps: int
    num_self_play_games: int
    optimizer: str
    data_buffer_size: int
    checkpoint_interval: int
    evaluation_games: int
    max_moves: int


@dataclass
class NetworkConfig:
    num_residual_blocks: int
    num_filters: int


@dataclass
class TemperatureConfig:
    initial_moves: int


@dataclass
class SelfPlayConfig:
    num_simulations: int
    temperature: TemperatureConfig
    resign_threshold: float
    disable_resignation_fraction: float


@dataclass
class EvaluationConfig:
    tournament_games: int
    time_per_move: int  # seconds
    evaluation_threshold: float


@dataclass
class Config:
    training: TrainingConfig
    network: NetworkConfig
    self_play: SelfPlayConfig
    evaluation: EvaluationConfig


def load_config(file_path: str = 'config.yaml') -> Config:
    """
    Loads the configuration from a YAML file and parses it into structured data classes.

    Args:
        file_path (str): Path to the YAML configuration file. Defaults to "config.yaml".

    Returns:
        Config: A structured configuration object containing all the settings.
    """
    with open(file_path, "r") as file:
        raw_config = yaml.safe_load(file)

    return Config(
        training=TrainingConfig(
            learning_rate=LearningRateConfig(**raw_config["training"]["learning_rate"]),
            batch_size=raw_config["training"]["batch_size"],
            momentum=raw_config["training"]["momentum"],
            weight_decay=raw_config["training"]["weight_decay"],
            loss_weights=LossWeightsConfig(**raw_config["training"]["loss_weights"]),
            training_steps=raw_config["training"]["training_steps"],
            num_self_play_games=raw_config["training"]["num_self_play_games"],
            optimizer=raw_config["training"]["optimizer"],
            data_buffer_size=raw_config["training"]["data_buffer_size"],
            checkpoint_interval=raw_config["training"]["checkpoint_interval"],
            evaluation_games=raw_config["training"]["evaluation_games"],
            max_moves=raw_config["training"]["max_moves"]
        ),
        network=NetworkConfig(**raw_config["network"]),
        self_play=SelfPlayConfig(
            num_simulations=raw_config["self_play"]["num_simulations"],
            temperature=TemperatureConfig(**raw_config["self_play"]["temperature"]),
            resign_threshold=raw_config["self_play"]["resign_threshold"],
            disable_resignation_fraction=raw_config["self_play"]["disable_resignation_fraction"],
        ),
        evaluation=EvaluationConfig(**raw_config["evaluation"]),
    )