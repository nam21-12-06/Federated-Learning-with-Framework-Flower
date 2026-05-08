# Federated Learning Research Framework (Flower + PyTorch)

## Overview

This project implements a modular, configuration-driven Federated Learning (FL) framework using Flower and PyTorch.

It is designed for research on:

- Aggregation algorithms
- Robust federated optimization
- Byzantine-resilient learning under adversarial attacks

The framework emphasizes:

- **Clean Architecture**
- **Extensibility**
- **Reproducibility**
- **Research-oriented experimentation**

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

- `n` = total number of clients
- `f` = number of Byzantine clients

---

## Attack Simulation

### Sign-Flip Attack

Implemented Byzantine attack:

- Malicious clients multiply model updates by a negative scale factor

Example:

$$
w_{attack} = -w
$$

Purpose:

- Corrupt global aggregation
- Test robustness of FL aggregation algorithms

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

The dataset is automatically downloaded using `torchvision` when running the client for the first time.

No manual download is required.

By default, data is stored in:

```text
data/
└── cifar-10-batches-py/
```

---

## Data Distribution

### IID

- Random uniform split across clients

---

### Label-Skew

- Each client only receives a subset of labels

Example:

- Client 0 → airplane, automobile
- Client 1 → bird, cat

Used to simulate heterogeneous local datasets.

---

### Dirichlet Non-IID

Simulates realistic non-IID distributions.

Controlled by parameter:

```yaml
dirichlet_alpha
```

Interpretation:

- Small alpha → highly skewed
- Large alpha → closer to IID

Example:

```yaml
dataset:
  partition_type: dirichlet
  dirichlet_alpha: 0.5
```

---

## Model

Simple CNN architecture:

- Conv2D(3 → 32)
- Conv2D(32 → 64)
- MaxPooling
- Fully Connected Layer (128)
- Output Layer (10 classes)

---

## Configuration-Driven Architecture

This project uses YAML configuration files to ensure:

- Reproducibility
- Cleaner experiments
- Easy parameter management

The architecture intentionally separates:

- **Server configuration**
- **Client configuration**

to better mimic real Federated Learning systems.

---

## Example Configuration

### Server Config

```yaml
server:
  strategy: krum
  rounds: 5
  min_clients: 5
```

---

### Client Config

```yaml
client:
  num_clients: 5
  batch_size: 32
  local_epochs: 1
  learning_rate: 0.001
```

---

### Attack Config

```yaml
attack:
  enabled: true
  type: signflip
  byzantine_ratio: 0.2

  params:
    scale: -1.0
```

---

## Project Structure

```text
project/

├── aggregators/
│   └── krum.py

├── attacks/
│   ├── base.py
│   └── sign_flip.py

├── configs/
│   ├── fedavg.yaml
│   └── krum.yaml

├── core/
│   └── config.py

├── strategies/
│   ├── strategy_factory.py
│   └── krum_strategy.py

├── utils/
│   ├── history.py
│   └── plotting.py

├── data/
├── results/

├── client.py
├── server.py
├── dataset.py
├── model.py
└── run_experiment.py
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

## Method 1 — Automated Experiment Runner

Recommended for research experiments.

Run:

```bash
python run_experiment.py
```

The script automatically:

- Starts the server
- Launches clients
- Waits for completion
- Saves results
- Cleans up processes

---

## Method 2 — Manual Execution

Useful for debugging.

---

### Step 1 — Run Server

FedAvg:

```bash
python server.py configs/fedavg.yaml
```

Krum:

```bash
python server.py configs/krum.yaml
```

---

### Step 2 — Run Clients

Open multiple terminals:

```bash
python client.py configs/fedavg.yaml --partition-id 0
```

```bash
python client.py configs/fedavg.yaml --partition-id 1
```

```bash
python client.py configs/fedavg.yaml --partition-id 2
```

```bash
python client.py configs/fedavg.yaml --partition-id 3
```

```bash
python client.py configs/fedavg.yaml --partition-id 4
```

---

## CLI Override Support

Configuration values can also be overridden directly from CLI.

Example:

```bash
python server.py configs/fedavg.yaml --rounds 10
```

Priority:

```text
CLI arguments > YAML config
```

---

## Output & Results

Results are automatically saved in:

```text
results/
```

Generated files include:

```text
history_<strategy>.json
results_<strategy>.png
```

---

### JSON History

Contains:

- Loss history
- Accuracy history
- Round-by-round metrics

Useful for:

- Research analysis
- Plotting
- Benchmarking
- Reproducibility

---

### Result Plots

Automatically generated graphs:

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

- Similar scores → consistent clients
- Large outlier → possible Byzantine behavior

---

## Expected Behavior

### Without Attack

- FedAvg → stable
- FedProx → stable
- Krum → similar to FedAvg

---

### With Byzantine Attack

- FedAvg → vulnerable
- Krum → more robust
- FedProx → not Byzantine-robust by itself

---

### With Non-IID Data

- FedProx → improved stability
- Krum → may slightly degrade

---

## Current Features

- FedAvg
- FedProx
- Krum
- IID partitioning
- Label-skew partitioning
- Dirichlet Non-IID
- Sign-Flip attack
- Distributed evaluation
- Metric aggregation
- YAML configuration system
- CLI override support
- Experiment automation
- Result plotting
- JSON history saving

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
- Logging system

---

## Notes

This framework is intended for:

- Research
- Experimentation
- Robust FL benchmarking

It is **not optimized for production deployment**.

---

## Author

Nguyễn Thái Hoài Nam

Federated Learning Project
