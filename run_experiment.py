import subprocess
import time
import sys
from core.config import load_config

def run_experiment(server_config_path, client_config_path="configs/client_config.yaml"):
    server_cfg = load_config(server_config_path)
    client_cfg = load_config(client_config_path)

    experiment_name = server_cfg.get("experiment_name", "fl_experiment")
    print(f"\n===== Running {experiment_name} =====")

    # 1. Start Server with its own config file
    print("-> Starting Server...")
    server_process = subprocess.Popen([
        sys.executable, "server.py", server_config_path
    ])
    
    # Wait 3 seconds for Server to setup
    time.sleep(3)

    # 2. Start Clients
    # Get num_clients from Client config file
    num_clients = client_cfg["dataset"].get("num_clients", 5)
    client_processes = []

    for i in range(num_clients):
        print(f"-> Starting Client {i}...")
        cmd = [
            sys.executable, "client.py",
            client_config_path,
            "--partition-id", str(i)
        ]
        p = subprocess.Popen(cmd)
        client_processes.append(p)
        time.sleep(1) # Sleep for 1s to avoid bottleneck during data loading

    # 3. Wait for training to finish
    try:
        server_process.wait()
    except KeyboardInterrupt:
        print("\n Stopping the system...")
    finally:
        # Clean up processes
        server_process.terminate()
        for p in client_processes:
            p.terminate()

    print(f"===== Finished {experiment_name} =====")


if __name__ == "__main__":
    # Run your experiments sequentially here
    run_experiment("configs/server_config_krum.yaml")
    
    # Uncomment the lines below to run Krum automatically after FedAvg
    # time.sleep(5) 
    # run_experiment("configs/server_config_krum.yaml")