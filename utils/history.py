import json


def save_history(history, strategy_name):

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

    with open(
        f"history_{strategy_name}.json",
        "w"
    ) as f:
        json.dump(data, f, indent=4)