import flwr as fl 
import torch
from torch.utils.data import DataLoader
from model import Net

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainset, testset):
        self.model = Net().to(DEVICE)

        self.testloader = DataLoader(testset, batch_size=32, shuffle=False)
        self.trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

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

        local_epochs = config.get("local_epochs", 1)
        lr = config.get("lr", 0.001)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        self.model.train()
        total_loss = 0.0

        # Train with local epoch 
        for epoch in range(local_epochs):
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

        # Return updated weights, number of samples, metrics (optional)
        return self.get_parameters(config), len(self.trainloader.dataset), {"train_loss": avg_loss}
    
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
    from dataset import (load_datasets, load_datasets_label_skew)

    # Parse config
    parser = argparse.ArgumentParser(description="Run Flower Client")
    parser.add_argument("--partition-id", type=int, required=True)
    parser.add_argument("--partition-type", type=str, default="iid", choices=["iid","label_skew"])
    parser.add_argument("--num-clients", type=int, default=2)
    args = parser.parse_args()

    print("Downloading data...", flush=True)
    if args.partition_type=="iid":
        client_trainsets,client_testsets = load_datasets(args.num_clients)
    else:
        client_trainsets,client_testsets = load_datasets_label_skew(args.num_clients)
    
    trainset = client_trainsets[args.partition_id]
    testset = client_testsets[args.partition_id]

    # Init and connect to server
    print(f"Connecting Client {args.partition_id} to server...", flush=True)
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(trainset, testset).to_client(),
    )