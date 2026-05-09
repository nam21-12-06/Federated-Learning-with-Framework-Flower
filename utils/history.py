import os
import json

def save_history(history, experiment_name, save_dir="results/"):
    data = {
        "loss": [
            (int(r), float(v))
            for r, v in history.losses_distributed
        ],
        "accuracy": [
            (int(r), float(v))
            for r, v in history.metrics_distributed.get(
                "accuracy", []
            )
        ]
    }

    os.makedirs(save_dir, exist_ok=True)
    
    file_path = os.path.join(save_dir, f"history_{experiment_name}.json")

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
        
    print(f"History saved: {file_path}")