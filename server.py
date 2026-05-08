import argparse
import flwr as fl

from core.config import load_config
from strategies.strategy_factory import build_strategy
from utils.plotting import plot_metrics
from utils.history import save_history

def parse_args():
    parser = argparse.ArgumentParser(description="Run Flower Server")

    # Config file
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML config file (e.g., configs/server_config_krum.yaml)"
    )

    # Optional CLI overrides
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--min_clients", type=int)
    parser.add_argument("--strategy", type=str)

    return parser.parse_args()

def apply_overrides(cfg, args):

    if args.rounds is not None:
        cfg["server"]["rounds"] = args.rounds
    if args.min_clients is not None:
        cfg["server"]["min_clients"] = args.min_clients
    if args.strategy is not None:
        cfg["server"]["strategy"] = args.strategy
    return cfg

def main():
    args = parse_args()
    
    # Load YAML config
    cfg = load_config(args.config)

    # Apply CLI override
    cfg = apply_overrides(cfg, args)

    strategy_name = cfg['server']['strategy']
    rounds = cfg['server']['rounds']
    min_clients = cfg['server']['min_clients']

    # Strategy
    strategy = build_strategy(strategy_name,
                            min_clients,
                            **cfg.get("strategy_params", {}))

    print(f"Strategy: {strategy_name.upper()} | Rounds: {rounds}")

    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )
    # Save history
    output_cfg = cfg.get('output', {})
    if output_cfg.get('save_history', False):
        save_history(history, strategy_name)

    if output_cfg.get('save_plot', False):
        plot_metrics(history, strategy_name)


if __name__ == "__main__":
    main()
