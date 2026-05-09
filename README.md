# Federated Learning Research Framework (Flower + PyTorch)

## Overview

This project implements a modular, configuration-driven Federated Learning (FL) framework using Flower and PyTorch.

It is designed for research on:

- Aggregation algorithms
- Robust federated optimization
- Byzantine-resilient learning under adversarial attacks

The framework emphasizes:

- **Clean Architecture:** Strict separation between Server and Client configurations
- **Extensibility:** Easy integration of new aggregation strategies and attacks
- **Reproducibility:** YAML-based experiment management
- **Research-oriented experimentation:** Automated scripting and dynamic result tracking

---

## What is Federated Learning?

Federated Learning (FL) is a distributed machine learning paradigm where:

- Clients train locally on private data
- Raw data never leaves client devices
- Only model updates are shared
- A central server aggregates updates

### Workflow

1. Server → send global model
2. Clients → train locally
3. Clients → send updates
4. Server → aggregate
5. Repeat for multiple rounds

---

## Implemented Strategies

### FedAvg

Standard Federated Averaging algorithm:

$$
w^{(t+1)} = \sum \left(\frac{n_k}{N}\right) w_k
$$

---

### FedProx

Extension of FedAvg to better handle heterogeneous (Non-IID) data:

$$
F_k(w) + \frac{\mu}{2}\|w - w_{global}\|^2
$$

---

### Krum

Byzantine-robust aggregation algorithm.

Instead of averaging all updates:

- Computes pairwise distances between client updates
- Selects the most reliable update

Robustness condition:

$$
n \ge 2f + 3
$$

Where:

- `n` = total clients
- `f` = Byzantine clients

---

## Attack Simulation

### Sign-Flip Attack

Malicious clients reverse the optimization direction:

$$
w_{attack} = -1.0 \times w
$$

Effects:

- Corrupts global aggregation
- Causes divergence under FedAvg
- Useful for testing Byzantine robustness

---

### Gaussian Attack

Malicious clients send random Gaussian noise instead of meaningful updates.

Gaussian noise:

$$
\mathcal{N}(\mu, \sigma^2)
$$

Example:

$$
w_{attack} = w + \mathcal{N}(0.0, 0.5)
$$

Effects:

- Destabilizes aggregation
- Slows convergence
- Simulates noisy Byzantine behavior

---

## Dataset Setup

Dataset used:

- **CIFAR-10**

Classes:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

The dataset is automatically downloaded using `torchvision`.

By default:

```text
data/
└── cifar-10-batches-py/
```

No manual download is required.

---

## Data Distribution

### IID (Default)

- Random uniform split across clients

---

### Label-Skew

Each client only receives a subset of labels.

Example:

- Client 0 → airplanes, cars
- Client 1 → cats, dogs

Used to simulate heterogeneous local datasets.

---

### Dirichlet Non-IID

Realistic non-IID distribution controlled by:

```yaml
dirichlet_alpha
```

Behavior:

- small alpha → highly skewed
- large alpha → closer to IID

---

## Model

Simple CNN architecture intentionally kept lightweight for rapid experimentation.

Architecture:

- Conv2D(3 → 32)
- Conv2D(32 → 64)
- MaxPooling
- Fully Connected Layer (128)
- Output Layer (10 classes)

---

## Configuration-Driven Architecture

This framework uses separated YAML configuration files.

### Server Config

Controls:

- aggregation strategy
- communication rounds
- aggregation parameters
- result saving

Example:

```yaml
server:
  strategy: krum
  rounds: 5
  min_clients: 5

strategy_params:
  f: 1

output:
  save_history: true
  save_plot: true
  save_dir: results/
```

---

### Client Config

Controls:

- local training
- dataset partitioning
- attack behavior

Example:

```yaml
client:
  batch_size: 32
  local_epochs: 1
  learning_rate: 0.001

dataset:
  name: cifar10
  num_clients: 5
  partition_type: iid

attack:
  enabled: true
  type: gaussian
  byzantine_ratio: 0.2

  params:
    mean: 0.0
    std: 0.5
```

---

## Project Structure

```text
project/

├── aggregators/          # Aggregation math and logic
├── attacks/              # Byzantine attack simulations
├── configs/              # YAML configuration files
├── core/                 # Core utilities
├── strategies/           # FL strategy implementations
├── utils/                # Plotting and history saving

├── data/                 # Auto-downloaded datasets
├── results/              # Experiment outputs

├── client.py             # FL Client
├── server.py             # FL Server
├── dataset.py            # Dataset partitioning
├── model.py              # CNN architecture
└── run_experiment.py     # Automated experiment runner
```

---

## Installation

Create virtual environment:

```bash
python -m venv fl_env
```

Activate environment (Windows):

```bash
fl_env\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

## Method 1 — Automated Runner

Run experiments automatically:

```bash
python run_experiment.py
```

The script automatically:

- launches server
- launches all clients
- saves plots
- saves JSON histories
- terminates processes safely

---

## Method 2 — Manual Execution

### Step 1 — Start Server

```bash
python server.py configs/server_config_fedavg.yaml \
--experiment-name fedavg_gaussian_attack
```

---

### Step 2 — Start Clients

Open multiple terminals:

```bash
python client.py configs/client_gaussian.yaml --partition-id 0

python client.py configs/client_gaussian.yaml --partition-id 1

python client.py configs/client_gaussian.yaml --partition-id 2

python client.py configs/client_gaussian.yaml --partition-id 3

python client.py configs/client_gaussian.yaml --partition-id 4
```

### Method 3 — CLI Overrides (Advanced Manual Execution)

The framework supports a **Hybrid Configuration System**. You can override specific YAML parameters directly from the command line for rapid testing and debugging without modifying the configuration files.

**Priority Rule:** `CLI Arguments > YAML Config`

**Example: Overriding Server Parameters**
If your `server_config_fedavg.yaml` defines 5 rounds, but you want to quickly test it for 10 rounds with 3 clients, you can override these values directly:

````bash
python server.py configs/server_config_fedavg.yaml --rounds 10 --min_clients 3 --experiment-name fedavg_10_rounds_quick_test
---

## Output & Results

Results are automatically saved in:

```text
results/
````

Generated files:

```text
results/history_fedavg_gaussian_attack.json

results/results_fedavg_gaussian_attack.png
```

---

## JSON History

Contains:

- loss history
- accuracy history
- round-by-round metrics

Useful for:

- benchmarking
- custom plotting
- research analysis

---

## Result Plots

Automatically generated:

- Loss vs Rounds
- Accuracy vs Rounds

---

## Debugging Krum

Example logs:

```text
[Krum] scores: [12.1, 11.9, 12.3, 12.0, 12.2]

[Krum] selected client 1
```

Interpretation:

- Similar scores → honest clients
- Large outlier → possible Byzantine client

---

## Expected Behavior

### Without Attack

- FedAvg → stable
- FedProx → stable
- Krum → similar to FedAvg

---

### Under Byzantine Attack

#### FedAvg

- highly vulnerable
- unstable convergence
- fluctuating loss

#### Krum

- more robust
- filters malicious updates
- restores stability

#### FedProx

- helps Non-IID training
- not Byzantine-robust by itself

---

## Current Features

- FedAvg
- FedProx
- Krum
- Sign-Flip attack
- Gaussian attack
- IID partitioning
- Label-skew partitioning
- Dirichlet Non-IID partitioning
- YAML configuration system
- Automated experiment runner
- JSON history saving
- Plot generation

---

## Planned Features

### Robust Aggregation

- Multi-Krum
- Trimmed Mean
- Bulyan
- FLTrust

---

### Attack Types

- Backdoor attack
- Label-flipping attack
- Adaptive Byzantine attack
- Gradient poisoning

---

### Engineering Improvements

- TensorBoard integration
- Checkpoint saving
- Docker support
- Multi-GPU training
- Experiment tracking

---

## Notes

This framework is intended for:

- research
- experimentation
- robust FL benchmarking

It is not optimized for production deployment.

---

## Author

Nguyễn Thái Hoài Nam

Federated Learning Research Project
