import flwr as fl 
import torch
from torch.utils.data import DataLoader
from model import Net
from attacks.sign_flip import SignFlipAttack
from attacks.gaussian import GaussianAttack
from core.config import load_config


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainset, testset, batch_size, local_epochs, lr, attack=None):
        self.model = Net().to(DEVICE)

        self.testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        self.trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        self.local_epochs = local_epochs
        self.lr = lr
        self.attack = attack

    # Send update to server
    def get_parameters(self, config):
        # Convert Tensor -> Numpy (Flower works with Numpy)
        # Numpy runs in CPU
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    # Receive model from server
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        # Convert Numpy -> Tensor
        state_dict = {k: torch.tensor(v).to(DEVICE) for k, v in params_dict}
        # Overwrite local model
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Local Training"""

        # Receive model from server
        self.set_parameters(parameters)

        # Using local parameters (lr, local_epochs)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        self.model.train()
        total_loss = 0.0

        # Train with local epoch 
        for epoch in range(self.local_epochs):
            for x, y in self.trainloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                loss = loss_fn(self.model(x), y)
                loss.backward()
                optimizer.step()
                # Train loss
                total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(self.trainloader.dataset)

        print(f"[Client] Train done | Loss: {avg_loss:.4f}", flush=True)

        params = self.get_parameters(config)

        # Use when activate attack
        if self.attack is not None:
            # Debug
            print("[Client] Byzantine attack applied", flush=True)
            
            params = self.attack.apply(params)

        # Return updated weights, number of samples, metrics (optional)
        return params, len(self.trainloader.dataset), {"train_loss": avg_loss}
    
    def evaluate(self, parameters, config):
        """Local Evaluation"""
        self.set_parameters(parameters)
        loss_fn = torch.nn.CrossEntropyLoss()
        self.model.eval()    # Switch to evaluation mode

        correct, total, total_loss = 0, 0, 0.0

        with torch.no_grad():
            # Use testloader 
            for x, y in self.testloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                outputs = self.model(x)
                
                loss = loss_fn(outputs, y)
                total_loss += loss.item() * x.size(0)

                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        accuracy = correct / total
        avg_loss = total_loss / total
        
        print(f"[Client] Eval | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}", flush=True)
        return float(avg_loss), total, {"accuracy": accuracy}
    
if __name__ == "__main__":
    import argparse
    from dataset import (load_datasets, 
                        load_datasets_label_skew,
                        load_datasets_dirichlet)

    # Parse config
    parser = argparse.ArgumentParser(description="Run Flower Client")
    parser.add_argument("config", type =str, help="Path to YAML config file")
    parser.add_argument("--partition-id", type=int, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    num_clients = cfg["dataset"].get("num_clients", 5)

    partition_type = cfg["dataset"]["partition_type"]
    dirichlet_alpha = cfg["dataset"].get("dirichlet_alpha", 0.5)

    batch_size = cfg["client"]["batch_size"]
    local_epochs = cfg["client"]["local_epochs"]
    lr = cfg["client"]["learning_rate"]

    attack = None
    if cfg["attack"].get("enabled", False):
        attack_type = cfg["attack"]["type"]
        byzantine_ratio = cfg["attack"]["byzantine_ratio"]
        num_byzantine = int(num_clients * byzantine_ratio)
        is_byzantine = args.partition_id < num_byzantine

        # Sign flip attack
        if is_byzantine and attack_type == "signflip":
            scale = cfg["attack"]["params"]["scale"]
            attack = SignFlipAttack(scale=scale)
            print(f"[Client {args.partition_id}] Byzantine SignFlip enabled", flush=True)
        # Gaussian attack
        elif attack == "gaussian":
            mean = cfg["attack"]["params"].get("mean", 0.0)
            std = cfg["attack"]["params"].get("std", 1.0)
            attack = GaussianAttack(mean=mean, std=std)
            print(f"[Client {args.partition_id}] Byzantine Gaussian Attack enabled (mean={mean}, std={std})", flush=True)


    print("Downloading data...", flush=True)
    if partition_type == "iid":
        client_trainsets, client_testsets = load_datasets(num_clients)
    elif partition_type == "label_skew":
        client_trainsets, client_testsets = load_datasets_label_skew(num_clients)
    else:
        client_trainsets, client_testsets = load_datasets_dirichlet(num_clients, alpha=dirichlet_alpha)

    trainset = client_trainsets[args.partition_id]
    testset = client_testsets[args.partition_id]

    # Init and connect to server
    print(f"Connecting Client {args.partition_id} to server...", flush=True)
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(trainset=trainset, 
            testset=testset, 
            batch_size=batch_size,
            local_epochs=local_epochs,
            lr=lr,
            attack=attack).to_client(),
    )