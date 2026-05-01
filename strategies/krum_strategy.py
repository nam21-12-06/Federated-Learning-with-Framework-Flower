from flwr.server.strategy import FedAvg
from aggregators.krum import krum, flatten_weights
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters


class KrumStrategy(FedAvg):

    def __init__(self, f=1, **kwargs):
        # f: Number of Byzantine clients 
        super().__init__(**kwargs)
        self.f = f
        self.latest_params = None       # Store global model in the previous round

    def aggregate_fit(self, server_round, results, failures):
        # Argument:
        # results: list client responses 
        # failure: number of Byzantine clients

        if not results:
            return None, {}
        
        if failures:
            print(f"[Krum] {len(failures)} failures")

        # ---- Extract client weights ----
        weights = [
            parameters_to_ndarrays(res.parameters)
            for _, res in results
        ]
        # ---- Get global model (previous round) ----
        if self.latest_params is None:
            base_weights = weights[0]
        else:
            base_weights = parameters_to_ndarrays(self.latest_params)

        base_vector = flatten_weights(base_weights)

        # ---- Compute updates ----
        # update = w_i - w_global
        updates = [flatten_weights(w) - base_vector
                for w in weights]
        
        # ---- Run Krum on updates ----
        idx, scores = krum(updates, self.f)

        print(f"[Krum] scores: {scores}")
        print(f"[Krum] selected client {idx} with score {scores[idx]}")

        # ---- Select corresponding WEIGHTS ----
        # Krum choose best client 
        selected_weights = weights[idx]

        # ---- Save new global model ----
        self.latest_parameters = ndarrays_to_parameters(selected_weights)

        # ---- Aggregate metrics ----
        metrics = {}
        if self.fit_metrics_aggregation_fn:
            metrics = self.fit_metrics_aggregation_fn(
                [(res.num_examples, res.metrics) for _, res in results]
            )

        return self.latest_parameters, metrics