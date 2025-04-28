from utils.networks import DemoNet
from azure.storage.blob import BlobServiceClient
import io
from utils.self_play import SelfPlaySession
from utils.agent import Agent
from utils.losses import combined_loss
from utils.memory import ReplayMemory
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Union, Literal, Optional
import json
import os
import torch.multiprocessing as mp
from copy import deepcopy
from dotenv import load_dotenv
import pickle


def train_on_batch(
    data: ReplayMemory, 
    network: DemoNet, 
    batch_size: int, 
    device: torch.device, 
    optimizer: Union[optim.Adam, optim.SGD],
    policy_weight: float = 1.0,
    value_weight: float = 1.0
) -> bool:
    network.to(device)
    network.train()
    if len(data) <= batch_size * 2:
        return False
    # Zero grad
    optimizer.zero_grad()
    # Get batch
    batch_dict = data.sample_in_batches(batch_size=batch_size)
    state_batch = batch_dict['state_batch'].to(device)
    policy_batch = batch_dict['policy_batch'].to(device)
    reward_batch = batch_dict['reward_batch'].to(device)

    # Forward pass
    policy_out, value_out = network(state_batch)
    loss = combined_loss(
        pi=policy_batch, 
        p_theta_logits=policy_out, 
        z=reward_batch, 
        v_theta=value_out, 
        policy_weight=policy_weight, 
        value_weight=value_weight
    )

    # Backward pass
    loss.backward()
    optimizer.step()
    return True


class Checkpoint:
    def __init__(self, verbose: bool):
        # self.compute_elo = compute_elo
        self.verbose = verbose
        self.best_agent = None
        self.best_weights = None
        self.best_model = None
        self.best_path = 'checkpoints/best_weights'
        self.iteration = -1  # Start with an invalid iteration
        self.best_version = -1
        load_dotenv(override=True)
        
        # 🔹 Azure Storage Configuration
        self.connection_string = os.environ.get("BLOB_CONNECTION_STRING")
        self.container_name = "dem0"

        # 🔹 Initialize Azure Blob Storage client
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        self.container_client = self.blob_service_client.get_container_client(self.container_name)
    def upload_to_blob(self, blob_folder: str, filename: str, data: bytes):
        """
        Uploads binary data to Azure Blob Storage inside a specific folder.
        """
        blob_name = f"{blob_folder}/{filename}"  # Store inside a subfolder (e.g., checkpoints/iteration_X/)
        blob_client = self.container_client.get_blob_client(blob_name)
        blob_client.upload_blob(data, overwrite=True)
        if self.verbose:
            print(f"✅ Uploaded {filename} to Azure Blob Storage at {blob_name}")

    def save_file(self, obj: any, blob_folder: str, filename: str):
        buffer = io.BytesIO()
        pickle.dump(obj, buffer)
        buffer.seek(0)
        self.upload_to_blob(blob_folder, filename, buffer.getvalue())

    def save_pretrained(self, state_dict: dict, folder: str = None):
        if not folder:
            blob_folder = "checkpoints"  # Folder structure
        else:
            blob_folder = f'checkpoints/{folder}'

        # 🔹 Convert PyTorch model state_dict to bytes
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        buffer.seek(0)

        # 🔹 Upload to Azure Blob Storage
        self.upload_to_blob(blob_folder, "pretrained_weights.pth", buffer.getvalue())

    def save_log(self, iteration: Union[int, Literal['best']]):
        """
        Uploads the `dem0.log` file to Azure Blob Storage.
        """
        if iteration == 'best':
            blob_folder = self.best_path
        else:
            blob_folder = f"checkpoints/iteration_{iteration}"  # Folder structure
        
        log_filename = "dem0.log"  # Log file name

        # Check if log file exists
        if not os.path.exists(log_filename):
            print(f"❌ Log file {log_filename} not found.")
            return
        
        # Read the log file
        with open(log_filename, "rb") as log_file:
            log_data = log_file.read()

        # Upload log file to Azure Blob Storage
        self.upload_to_blob(blob_folder, log_filename, log_data)

        if self.verbose:
            print(f"✅ Uploaded {log_filename} to Azure Blob Storage at {blob_folder}/{log_filename}")

    def save_state_dict(self, iteration: Union[int, Literal['best']], state_dict: Optional[dict] = None) -> None:
        """
        Saves model weights (`.pth`) to Azure Blob Storage inside a structured folder.
        """
        if iteration == 'best':
            blob_folder = self.best_path
        else:
            blob_folder = f"checkpoints/iteration_{iteration}"  # Folder structure
        # 🔹 Convert PyTorch model state_dict to bytes
        buffer = io.BytesIO()
        target_state = state_dict if state_dict is not None else self.best_weights
        torch.save(target_state, buffer)
        buffer.seek(0)
        # 🔹 Upload to Azure Blob Storage
        self.upload_to_blob(blob_folder, "weights.pth", buffer.getvalue())
    
    def save_model_obj(self, iteration: Union[int, Literal['best']], model: Optional[nn.Module] = None) -> None:
        """
        Saves the full PyTorch model (`.pth`) to Azure Blob Storage.
        """
        if iteration == 'best':
            blob_folder = self.best_path
        else:
            blob_folder = f"checkpoints/iteration_{iteration}"

        # 🔹 Convert model object to bytes
        buffer = io.BytesIO()
        torch.save(model if model is not None else self.best_model, buffer)
        buffer.seek(0)

        # 🔹 Upload to Azure Blob Storage
        self.upload_to_blob(blob_folder, "model.pth", buffer.getvalue())

    def save_info(self, info: dict, iteration: Union[int, Literal['best']]) -> None:
        """
        Saves metadata (`.json`) to Azure Blob Storage.
        """
        if iteration == 'best':
            blob_folder = self.best_path
        else:
            blob_folder = f"checkpoints/iteration_{iteration}"

        # 🔹 Convert JSON to bytes and upload
        json_bytes = json.dumps(info).encode('utf-8')
        self.upload_to_blob(blob_folder, "info.json", json_bytes)

    def save_replay_memory(self, memory: ReplayMemory, iteration: Union[int, Literal['best']]):
        """
        Saves the ReplayMemory object as a .pkl file in Azure Blob Storage.
        """
        if iteration == 'best':
            blob_folder = self.best_path
        else:
            blob_folder = f"checkpoints/iteration_{iteration}"

        # 🔹 Serialize ReplayMemory using pickle
        buffer = io.BytesIO()
        pickle.dump(list(memory.memory), buffer)
        buffer.seek(0)

        # 🔹 Upload to Azure Blob Storage
        self.upload_to_blob(blob_folder, "replay_memory.pkl", buffer.getvalue())

    def load_replay_memory(self, iteration: Union[int, Literal['best']]) -> list:
        """
        Loads the ReplayMemory object from Azure Blob Storage.
        """
        if iteration == 'best':
            blob_name = f"{self.best_path}/replay_memory.pkl"
        else:
            assert isinstance(iteration, int)
            blob_name = f"checkpoints/iteration_{iteration}/replay_memory.pkl"
        
        if not self.blob_exists(blob_name):
            raise FileNotFoundError(f"❌ ReplayMemory not found at {blob_name}")

        # 🔹 Download and deserialize ReplayMemory
        data = self.download_from_blob(blob_name)
        return data

    def save_best_stats(self, info: dict) -> None:
        self.save_model_obj(iteration="best")
        self.save_state_dict(iteration="best")
        self.save_info(info=info, iteration="best")
        self.save_log(iteration='best')

    def step(self, current_best_agent: Agent, info: dict, current_iteration: int):
        """
        Saves the best model, weights, replay memory, and metadata to Azure Blob Storage.
        """
        assert isinstance(current_best_agent, Agent), "Invalid agent provided"
        # Save the checkpoint for this iteration using the current agent's weights
        self.save_model_obj(iteration=current_iteration, model=current_best_agent.network)
        self.save_state_dict(iteration=current_iteration, state_dict=current_best_agent.network.state_dict())
        self.save_info(info=info, iteration=current_iteration)
        self.save_log(iteration=current_iteration)
        if self.verbose:
            print(f"✅ Checkpoint saved for iteration {current_iteration}")

        if current_best_agent.version > self.best_version:
            self.best_model = deepcopy(current_best_agent.network)
            self.best_agent = current_best_agent
            self.best_weights = self.best_model.state_dict()
            self.iteration = current_iteration  # Update iteration

            # 🔹 Upload to Azure
            self.save_best_stats(info=info)

            # if self.compute_elo:
            #     if self.verbose:
            #         print("📢 Starting background process for Elo computation...")

            #     # 🔹 Start Elo computation in a background process
            #     p = mp.Process(
            #         target=_background_compute_elo_and_save,
            #         args=(self.best_agent, f"{self.best_path}/info.json", self.verbose)
            #     )
            #     p.start()  # Runs in background
            # else:
            #     if self.verbose:
            #         print(f"✅ Best Weights have been checkpointed saved for iteration {self.iteration}")

    def download_from_blob(self, blob_name: str, device: torch.device = None) -> any:
        """
        Downloads and deserializes a blob (file) from Azure Blob Storage.

        Args:
            blob_name (str): Full blob path (e.g., 'checkpoints/iteration_3/weights.pth')
            device (torch.device): Device to load PyTorch tensors onto (if applicable)

        Returns:
            Any: The deserialized object (dict, model weights, Python object, etc.)

        Raises:
            ValueError: If file extension is not supported.
        """
        blob_client = self.container_client.get_blob_client(blob_name)
        blob_data = blob_client.download_blob().readall()
        if isinstance(blob_data, int):
            return pickle.load(blob_data)
        buffer = io.BytesIO(blob_data)

        # Detect file type and deserialize
        if blob_name.endswith((".pth", ".pt")):
            return torch.load(buffer, map_location=device)
        elif blob_name.endswith(".pkl"):
            return pickle.load(buffer)
        elif blob_name.endswith(".json"):
            return json.loads(blob_data.decode("utf-8"))
        else:
            raise ValueError(f"Unsupported file extension for blob: {blob_name}")

    def load_challenger_weights_if_available(self, iteration: int, network: nn.Module, logger, fallback_network: nn.Module = None) -> None:
        """
        Loads challenger network weights from Azure Blob Storage if available.
        Falls back to a provided network if challenger weights do not exist.

        Args:
            iteration (int): The iteration number to load challenger weights from.
            network (nn.Module): The network object to load the weights into.
            fallback_network (nn.Module, optional): Network to fall back to if no challenger weights found.
        """
        challenger_blob_path = f"checkpoints/iteration_{iteration}/weights.pth"
        
        if self.blob_exists(challenger_blob_path):
            # Challenger weights found, load them
            weights = self.download_from_blob(blob_name=challenger_blob_path)
            network.load_state_dict(weights)
            logger.info(f"✅ Loaded challenger weights from {challenger_blob_path}")
        else:
            # Challenger weights not found, fall back
            if fallback_network is not None:
                network.load_state_dict(fallback_network.state_dict())
                logger.info(f"⚠️ No challenger weights found at {challenger_blob_path}. Loaded fallback network weights.")
            else:
                logger.info(f"❌ No challenger weights found at {challenger_blob_path}, and no fallback network provided.")
    
    def blob_exists(self, blob_name: str) -> bool:
        """
        Checks if a blob (file) exists in Azure Blob Storage.
        
        Args:
            blob_name (str): The name (path) of the blob in the container.
            connection_string (str): Azure Blob Storage connection string.
            container_name (str): The name of the storage container.

        Returns:
            bool: True if the blob exists, False otherwise.
        """
        # Create a BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        container_client = blob_service_client.get_container_client(self.container_name)

        # Get blob client for the specific file
        blob_client = container_client.get_blob_client(blob_name)

        # Check if the blob exists
        return blob_client.exists()
    
    def list_folder_names(self) -> list:
        """
        Lists all folders (virtual directories) in the given blob path prefix.

        Args:
            prefix (str): The prefix path to look under (e.g., 'checkpoints/')

        Returns:
            List[str]: A list of folder names (as strings) under the prefix.
        """
        blobs = self.container_client.list_blobs()
        folder_names = set()

        for blob in blobs:
            parts = blob.name.split('/')
            for part in parts:
                if part.startswith("iteration_"):
                    folder_names.add(part)
                    break  # Once we find the folder name, no need to go further

        return sorted(folder_names)


# def _background_compute_elo_and_save(agent: Agent, info_blob_path: str, verbose: bool) -> None:
#     """
#     Computes Elo rating and updates metadata in Azure Blob Storage.
#     """
#     if verbose:
#         print("📢 Beginning Elo computation in background process...")

#     agent.compute_bayes_elo()  # Slow function

#     # 🔹 Prepare updated info
#     info = {"iteration": agent.iteration, "elo": agent.elo}

#     # 🔹 Convert JSON to bytes
#     json_bytes = json.dumps(info).encode('utf-8')

#     # 🔹 Upload updated JSON file
#     blob_service_client = BlobServiceClient.from_connection_string(os.environ.get("BLOB_CONNECTION_STRING"))
#     container_client = blob_service_client.get_container_client("checkpoints")
#     blob_client = container_client.get_blob_client(info_blob_path)
#     blob_client.upload_blob(json_bytes, overwrite=True)

#     if verbose:
#         print(f"✅ Finished Elo computation and saved updated info to {info_blob_path}")
