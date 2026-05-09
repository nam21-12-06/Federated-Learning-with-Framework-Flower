import subprocess
import time
import sys
from core.config import load_config

def run_experiment(server_config_path, client_config_path="configs/client_no_attack.yaml"):
    server_cfg = load_config(server_config_path)
    client_cfg = load_config(client_config_path)

    # 1. GENERATE DYNAMIC EXPERIMENT NAME
    strategy_name = server_cfg["server"]["strategy"]
    
    # Check if attack is enabled in client config safely
    is_attack_enabled = client_cfg.get("attack", {}).get("enabled", False)
    
    if is_attack_enabled:
        attack_type = client_cfg["attack"].get("type", "unknown_attack")
        experiment_name = f"{strategy_name}_{attack_type}_attack"
    else:
        experiment_name = f"{strategy_name}_no_attack"

    print(f"\n{'='*50}")
    print(f" RUNNING EXPERIMENT: {experiment_name.upper()}")
    print(f"{'='*50}")

    # 2. START SERVER WITH DYNAMIC EXPERIMENT NAME
    print("-> Starting Server...")
    server_process = subprocess.Popen([
        sys.executable, "server.py", 
        server_config_path,
        "--experiment-name", experiment_name  # <-- Passing the combined name here
    ])
    
    # Wait 3 seconds for Server to setup
    time.sleep(3)

    # 3. START CLIENTS
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

    # 4. WAIT FOR TRAINING TO FINISH
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

    # Scene 1 : FedAvg without attack (Baseline)
    # run_experiment(
    #     server_config_path="configs/server_config_fedavg.yaml", 
    #     client_config_path="configs/client_no_attack.yaml"
    # )

    # Scene 2 : FedAvg with attack Sign-Flip
    # run_experiment(
    #     server_config_path="configs/server_config_fedavg.yaml", 
    #     client_config_path="configs/client_signflip.yaml"
    # )

    # Scene 3 : FedAvg with attack Gaussian
    run_experiment(
        server_config_path="configs/server_config_fedavg.yaml", 
        client_config_path="configs/client_gaussian.yaml"
    )

    # Scene 4 : Krum defense Gaussian
    # run_experiment(
    #     server_config_path="configs/server_config_krum.yaml", 
    #     client_config_path="configs/client_gaussian.yaml"
    # )