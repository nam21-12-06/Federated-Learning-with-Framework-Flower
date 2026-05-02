import argparse
import flwr as fl
import matplotlib.pyplot as plt
from typing import List, Tuple
from flwr.common import Metrics
from strategies.strategy_factory import build_strategy
import json

# Use for script
def save_history(history, strategy_name):
    data = {
        "loss": [(int(r), float(v)) for r, v in history.losses_distributed],
        "accuracy": [
            (int(r), float(v)) 
            for r, v in history.metrics_distributed.get("accuracy", [])
        ]
    }

    with open(f"history_{strategy_name}.json", "w") as f:
        json.dump(data, f)


def plot_metrics(history, strategy_name):
    plt.figure(figsize=(12, 5))

    # ---- LOSS ----
    if len(history.losses_distributed) > 0:
        rounds_loss, values_loss = zip(*history.losses_distributed)

        plt.subplot(1, 2, 1)
        plt.plot(rounds_loss, values_loss, marker='o')
        plt.title(f"{strategy_name} - Loss")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.grid(True)

    # ---- ACCURACY ----
    if "accuracy" in history.metrics_distributed:
        rounds_acc, values_acc = zip(*history.metrics_distributed["accuracy"])

        plt.subplot(1, 2, 2)
        plt.plot(rounds_acc, values_acc, marker='o')
        plt.title(f"{strategy_name} - Accuracy")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"results_{strategy_name}.png")
    print(f"Results saved: results_{strategy_name}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, 
                        default="fedavg",
                        choices=["fedavg", "fedprox", "krum"])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--min_clients", type=int, default=2)   
    args = parser.parse_args()

    # Strategy
    strategy = build_strategy(
    args.strategy,
    args.min_clients
)

    print(f"Strategy: {args.strategy.upper()} | Rounds: {args.rounds}")

    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    # Optional 
    save_history(history, args.strategy)

    plot_metrics(history, args.strategy)


if __name__ == "__main__":
    main()

