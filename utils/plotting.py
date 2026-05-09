import os
import matplotlib.pyplot as plt

def plot_metrics(history, experiment_name, save_dir="results/"):
    plt.figure(figsize=(12, 5))

    # LOSS
    if len(history.losses_distributed) > 0:
        rounds_loss, values_loss = zip(
            *history.losses_distributed
        )

        plt.subplot(1, 2, 1)
        plt.plot(
            rounds_loss,
            values_loss,
            marker='o'
        )

        plt.title(f"{experiment_name} - Loss")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.grid(True)

    # ACCURACY
    if "accuracy" in history.metrics_distributed:
        rounds_acc, values_acc = zip(
            *history.metrics_distributed["accuracy"]
        )

        plt.subplot(1, 2, 2)
        plt.plot(
            rounds_acc,
            values_acc,
            marker='o'
        )

        plt.title(f"{experiment_name} - Accuracy")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.grid(True)

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    
    file_path = os.path.join(save_dir, f"results_{experiment_name}.png")

    plt.savefig(file_path)
    print(f"Results saved: {file_path}")
    
    plt.close()