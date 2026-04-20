from typing import List, Tuple
from flwr.common import Metrics


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    
    accuracies = [
        num_examples * m["accuracy"]
        for num_examples, m in metrics
        if "accuracy" in m
    ]

    examples = [
        num_examples
        for num_examples, m in metrics
        if "accuracy" in m
    ]

    if len(examples)==0:
        return {}

    return {
        "accuracy": sum(accuracies)/sum(examples)
    }