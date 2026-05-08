import flwr as fl
from aggregators.metrics import weighted_average
from strategies.krum_strategy import KrumStrategy

def build_strategy(name: str, min_clients: int, **kwargs):
    """
    Builds the federated learning strategy based on server configuration.
    All strategy-specific parameters (like 'f') are passed via **kwargs.
    """
    
    # Standard configuration for all strategies
    # Note: 'on_fit_config_fn' is REMOVED to respect client autonomy
    strategy_config = {
        "fraction_fit": 1.0,
        "fraction_evaluate": 1.0,
        "min_fit_clients": min_clients,
        "min_evaluate_clients": min_clients,
        "min_available_clients": min_clients,
        "fit_metrics_aggregation_fn": weighted_average,
        "evaluate_metrics_aggregation_fn": weighted_average,
    }

    strategy_name = name.lower()

    if strategy_name == "fedavg":
        return fl.server.strategy.FedAvg(**strategy_config)

    elif strategy_name == "fedprox":
        # Get mu from config or use default 1.0
        proximal_mu = kwargs.get("proximal_mu", 1.0)
        return fl.server.strategy.FedProx(
            **strategy_config,
            proximal_mu=proximal_mu
        )

    elif strategy_name == "krum":
        # Get 'f' dynamically from YAML (strategy_params)
        f_value = kwargs.get("f", 0) 
        print(f"-> Strategy Factory: Building Krum with f={f_value}")
        
        return KrumStrategy(
            **strategy_config,
            f=f_value
        )

    else:
        raise ValueError(f"Unknown strategy: {name}")