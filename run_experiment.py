import subprocess
import time
import json
import os
import signal
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
NUM_CLIENTS = 5
BYZANTINE_IDS = [0]   # có thể mở rộng nhiều client
ROUNDS = 5

STRATEGIES = ["fedavg", "krum"]

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)


# =========================
# PROCESS MANAGEMENT
# =========================
def run_server(strategy):
    return subprocess.Popen(
        [
            "python", "server.py",
            "--strategy", strategy,
            "--rounds", str(ROUNDS),
            "--min_clients", str(NUM_CLIENTS),
        ]
    )


def run_clients():
    processes = []

    for i in range(NUM_CLIENTS):
        cmd = [
            "python", "client.py",
            "--partition-id", str(i),
            "--num-clients", str(NUM_CLIENTS),
        ]

        if i in BYZANTINE_IDS:
            cmd += ["--attack", "signflip"]

        p = subprocess.Popen(cmd)
        processes.append(p)

    return processes


def wait_processes(processes):
    for p in processes:
        p.wait()


def shutdown_process(p):
    try:
        p.send_signal(signal.SIGINT)
        p.wait(timeout=10)
    except Exception:
        p.kill()


# =========================
# EXPERIMENT RUN
# =========================
def run_experiment(strategy):
    print(f"\n===== Running {strategy.upper()} =====")

    # ---- start server ----
    server = run_server(strategy)
    time.sleep(5)  # đợi server ổn định

    # ---- start clients ----
    clients = run_clients()

    # ---- wait clients ----
    wait_processes(clients)

    # ---- wait server finish ----
    server.wait()

    print(f"===== Finished {strategy.upper()} =====")


# =========================
# LOAD HISTORY
# =========================
def load_history(strategy):
    path = f"history_{strategy}.json"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")

    with open(path, "r") as f:
        return json.load(f)


# =========================
# PLOT COMPARISON
# =========================
def plot_compare():
    fedavg = load_history("fedavg")
    krum = load_history("krum")

    rounds_fa, acc_fa = zip(*fedavg["accuracy"])
    rounds_kr, acc_kr = zip(*krum["accuracy"])

    plt.figure(figsize=(8, 5))

    plt.plot(rounds_fa, acc_fa, marker='o', label="FedAvg")
    plt.plot(rounds_kr, acc_kr, marker='o', label="Krum")

    plt.title("FedAvg vs Krum (Byzantine: SignFlip)")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(RESULT_DIR, "comparison.png")
    plt.savefig(save_path)
    plt.show()

    print(f"Saved plot: {save_path}")


# =========================
# SAVE EXPERIMENT CONFIG
# =========================
def save_experiment_config():
    config = {
        "num_clients": NUM_CLIENTS,
        "byzantine_ids": BYZANTINE_IDS,
        "rounds": ROUNDS,
        "strategies": STRATEGIES,
        "attack": "signflip"
    }

    with open(os.path.join(RESULT_DIR, "experiment_config.json"), "w") as f:
        json.dump(config, f, indent=4)


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    save_experiment_config()

    for strategy in STRATEGIES:
        run_experiment(strategy)

    plot_compare()