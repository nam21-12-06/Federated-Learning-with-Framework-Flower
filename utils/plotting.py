import matplotlib.pyplot as plt


def plot_metrics(history, strategy_name):

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

        plt.title(f"{strategy_name} - Loss")
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

        plt.title(f"{strategy_name} - Accuracy")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        f"results_{strategy_name}.png"
    )

    print(
        f"Results saved: "
        f"results_{strategy_name}.png"
    )