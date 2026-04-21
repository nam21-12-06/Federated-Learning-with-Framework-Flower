import flwr as fl

from aggregators.metrics import weighted_average


def build_strategy(name, min_clients):

    strategy_config = {

        "fraction_fit":1.0,
        "fraction_evaluate":1.0,

        "min_fit_clients":min_clients,
        "min_evaluate_clients":min_clients,
        "min_available_clients":min_clients,

        "on_fit_config_fn": fit_config,

        "fit_metrics_aggregation_fn": weighted_average,
        "evaluate_metrics_aggregation_fn":
            weighted_average
    }

    if name=="fedavg":
        return fl.server.strategy.FedAvg(
            **strategy_config
        )

    elif name=="fedprox":
        return fl.server.strategy.FedProx(
            **strategy_config,
            proximal_mu=1.0
        )

    else:
        raise ValueError(
            f"Unknown strategy {name}"
        )
    

def fit_config(server_round):
    return {
        "local_epochs": 2,
        "lr": 0.001,
    }