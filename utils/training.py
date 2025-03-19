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
from typing import Union
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
):
    network.to(device)
    network.train()
    if len(data) <= batch_size:
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
    def __init__(self, verbose: bool, compute_elo: bool = True):
        self.compute_elo = compute_elo
        self.verbose = verbose
        self.best_agent = None
        self.best_weights = None
        self.best_model = None
        self.version = -1  # Start with an invalid version
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
        blob_name = f"{blob_folder}/{filename}"  # Store inside a subfolder (e.g., checkpoints/version_X/)
        blob_client = self.container_client.get_blob_client(blob_name)
        blob_client.upload_blob(data, overwrite=True)
        if self.verbose:
            print(f"✅ Uploaded {filename} to Azure Blob Storage at {blob_name}")

    def save_pretrained(self, state_dict: dict):
        blob_folder = "checkpoints"  # Folder structure

        # 🔹 Convert PyTorch model state_dict to bytes
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        buffer.seek(0)

        # 🔹 Upload to Azure Blob Storage
        self.upload_to_blob(blob_folder, "pretrained_weights.pth", buffer.getvalue())

    def save_log(self):
        """
        Uploads the `dem0.log` file to Azure Blob Storage.
        """
        blob_folder = "checkpoints"  # Folder structure
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

    def save_state_dict(self, path: str = None, state_dict: dict = None) -> None:
        """
        Saves model weights (`.pth`) to Azure Blob Storage inside a structured folder.
        """
        blob_folder = "checkpoints"  # Folder structure

        # 🔹 Convert PyTorch model state_dict to bytes
        buffer = io.BytesIO()
        if path and state_dict:
            torch.save(state_dict, buffer)
        else:
            torch.save(self.best_weights, buffer)

        buffer.seek(0)

        # 🔹 Upload to Azure Blob Storage
        if path is None:
            self.upload_to_blob(blob_folder, "weights.pth", buffer.getvalue())
        elif state_dict:
            self.upload_to_blob(blob_folder, path, buffer.getvalue())


    def save_model_obj(self) -> None:
        """
        Saves the full PyTorch model (`.pth`) to Azure Blob Storage.
        """
        blob_folder = f"checkpoints"

        # 🔹 Convert model object to bytes
        buffer = io.BytesIO()
        torch.save(self.best_model, buffer)
        buffer.seek(0)

        # 🔹 Upload to Azure Blob Storage
        self.upload_to_blob(blob_folder, "model.pth", buffer.getvalue())

    def save_info(self, info: dict) -> None:
        """
        Saves metadata (`.json`) to Azure Blob Storage.
        """
        assert isinstance(self.version, int), "Version must be an integer"
        blob_folder = f"checkpoints"

        # 🔹 Create metadata dictionary
        info['version'] = self.version

        # 🔹 Convert JSON to bytes and upload
        json_bytes = json.dumps(info).encode('utf-8')
        self.upload_to_blob(blob_folder, "info.json", json_bytes)

    def save_replay_memory(self, memory: ReplayMemory):
        """
        Saves the ReplayMemory object as a .pkl file in Azure Blob Storage.
        """
        blob_folder = f"checkpoints"

        # 🔹 Serialize ReplayMemory using pickle
        buffer = io.BytesIO()
        pickle.dump(list(memory.memory), buffer)
        buffer.seek(0)

        # 🔹 Upload to Azure Blob Storage
        self.upload_to_blob(blob_folder, "replay_memory.pkl", buffer.getvalue())

    def load_replay_memory(self) -> list:
        """
        Loads the ReplayMemory object from Azure Blob Storage.
        """
        blob_name = f"checkpoints/replay_memory.pkl"
        
        if not self.blob_exists(blob_name):
            raise FileNotFoundError(f"❌ ReplayMemory not found at {blob_name}")

        # 🔹 Download and deserialize ReplayMemory
        data = self.download_from_blob(blob_name)
        buffer = io.BytesIO(data)
        return pickle.load(buffer)

    def step(self, current_best_agent: Agent, memory: ReplayMemory, info: dict):
        """
        Saves the best model, weights, replay memory, and metadata to Azure Blob Storage.
        """
        assert isinstance(current_best_agent, Agent), "Invalid agent provided"
        
        if current_best_agent.version > self.version:
            self.best_model = deepcopy(current_best_agent.network)
            self.best_agent = current_best_agent
            self.best_weights = self.best_model.state_dict()
            self.version = current_best_agent.version  # Update version

            # 🔹 Upload to Azure
            self.save_model_obj()
            self.save_state_dict()
            self.save_info(info=info)
            self.save_replay_memory(memory)
            self.save_log()

            if self.compute_elo:
                if self.verbose:
                    print("📢 Starting background process for Elo computation...")
                
                # 🔹 Start Elo computation in a background process
                p = mp.Process(
                    target=_background_compute_elo_and_save,
                    args=(current_best_agent, f"checkpoints/info.json", self.verbose)
                )
                p.start()  # Runs in background
            else:
                if self.verbose:
                    print(f"✅ Checkpoint saved for version {self.version}")

    def download_from_blob(self, blob_name: str, device: torch.device = None, return_bytes: bool = True) -> any:
        """
        Downloads a file from Azure Blob Storage and returns its binary content.
        """
        blob_client = self.container_client.get_blob_client(blob_name)
        blob_data = blob_client.download_blob().readall()
        if device and not return_bytes:
            buffer = io.BytesIO(blob_data)
            return torch.load(buffer, map_location=device)
        return blob_data

    def load_best_checkpoint(self, network, save_local: bool = False, local_dir: str = "local_checkpoints/"):
        """
        Loads the model, weights, and metadata from Azure Blob Storage.

        Args:
            version (int): The model version to retrieve.
            network (torch.nn.Module): The PyTorch model instance to load weights into.
            save_local (bool): Whether to save the downloaded files locally.
            local_dir (str): Directory to save the local copy.

        Returns:
            dict: Metadata (e.g., version, elo).
        """

        # 🔹 Download and load weights
        weights_data = self.download_from_blob(f"checkpoints/weights.pth")
        weights_buffer = io.BytesIO(weights_data)
        network.load_state_dict(torch.load(weights_buffer))

        # 🔹 Download and load full model (if needed)
        model_data = self.download_from_blob(f"checkpoints/model.pth")
        model_buffer = io.BytesIO(model_data)
        model = torch.load(model_buffer)

        # 🔹 Download metadata
        json_data = self.download_from_blob(f"checkpoints/info.json")
        metadata = json.loads(json_data.decode("utf-8"))

        # 🔹 Save locally if requested
        if save_local:
            os.makedirs(local_dir, exist_ok=True)
            with open(os.path.join(local_dir, "weights.pth"), "wb") as f:
                f.write(weights_data)
            with open(os.path.join(local_dir, "model.pth"), "wb") as f:
                f.write(model_data)
            with open(os.path.join(local_dir, "info.json"), "w") as f:
                json.dump(metadata, f, indent=4)

        return metadata, model
    
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


def _background_compute_elo_and_save(agent: Agent, info_blob_path: str, verbose: bool) -> None:
    """
    Computes Elo rating and updates metadata in Azure Blob Storage.
    """
    if verbose:
        print("📢 Beginning Elo computation in background process...")

    agent.compute_bayes_elo()  # Slow function

    # 🔹 Prepare updated info
    info = {"version": agent.version, "elo": agent.elo}

    # 🔹 Convert JSON to bytes
    json_bytes = json.dumps(info).encode('utf-8')

    # 🔹 Upload updated JSON file
    blob_service_client = BlobServiceClient.from_connection_string(os.environ.get("BLOB_CONNECTION_STRING"))
    container_client = blob_service_client.get_container_client("checkpoints")
    blob_client = container_client.get_blob_client(info_blob_path)
    blob_client.upload_blob(json_bytes, overwrite=True)

    if verbose:
        print(f"✅ Finished Elo computation and saved updated info to {info_blob_path}")
