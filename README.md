# Federated Learning Research Framework (Flower + PyTorch)

## Overview

This project implements a modular Federated Learning (FL) framework using Flower and PyTorch, designed for research on:

- Aggregation algorithms
- Robust federated optimization
- Byzantine-resilient learning

The framework emphasizes:

- Clean architecture
- Extensibility
- Reproducibility

---

## What is Federated Learning?

Federated Learning (FL) is a distributed machine learning paradigm where:

- Clients train locally on private data
- Raw data never leaves client devices
- Only model updates are shared
- A central server aggregates updates

Workflow:

1. Server → send global model
2. Clients → train locally
3. Clients → send updates
4. Server → aggregate
5. Repeat for multiple rounds

---

## Implemented Strategies

### FedAvg

Standard Federated Averaging algorithm:

$$w^{(t+1)} = \sum \left(\frac{n_k}{N}\right) w_k$$

---

### FedProx

Extension of FedAvg to handle heterogeneous data:

$$F_k(w) + \frac{\mu}{2}\|w - w_{global}\|^2$$

---

### Krum

Byzantine-robust aggregation algorithm.
Instead of averaging:

- Computes pairwise distances between client updates
- Selects the most reliable update

Condition:

$$n \ge 2f + 3$$

Where:

- `n`: number of clients
- `f`: number of Byzantine clients

---

## Dataset

Dataset used:

- CIFAR-10

Classes:

- airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

### Data Distribution

**IID (default)**

- Random uniform split across clients

**Dirichlet non-IID**

- Simulates real-world data heterogeneity.
- Controlled by parameter `alpha`:
  - small alpha → highly skewed
  - large alpha → closer to IID

---

## Model

Simple CNN architecture:

- Conv2d(3 → 32)
- Conv2d(32 → 64)
- MaxPooling
- Fully Connected (128)
- Output layer (10 classes)

---

## Project Structure

```text
project/

├── server.py
├── client.py
├── model.py
├── dataset.py

├── strategies/
│   ├── strategy_factory.py
│   └── krum_strategy.py

├── aggregators/
│   └── krum.py

├── data/
└── fl_env/
```

---

## Installation

Create virtual environment:

```bash
python -m venv fl_env
```

Activate (Windows):

```bash
fl_env\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Run Server

FedAvg:

```bash
python server.py --strategy fedavg
```

FedProx:

```bash
python server.py --strategy fedprox
```

Krum:

```bash
python server.py --strategy krum --rounds 5 --min_clients 5
```

_Note: Krum requires $n \ge 2f + 3$_

---

### Run Clients

Example with 5 clients:

```bash
python client.py --partition-id 0 --num-clients 5
python client.py --partition-id 1 --num-clients 5
python client.py --partition-id 2 --num-clients 5
python client.py --partition-id 3 --num-clients 5
python client.py --partition-id 4 --num-clients 5
```

---

### Using Dirichlet Distribution

In `client.py`, replace dataset loading:

```python
load_datasets_dirichlet(num_clients, alpha=0.5)
```

Example:

```python
client_trainsets, client_testsets = load_datasets_dirichlet(
    args.num_clients,
    alpha=0.5
)
```

---

## Output

Results are saved as:

```text
results_<strategy>.png
```

Includes:

- Loss vs rounds
- Accuracy vs rounds

---

## Debugging Krum

Example logs:

```text
[Krum] scores: [12.1, 11.9, 12.3, 12.0, 12.2]
[Krum] selected client 1
```

Interpretation:

- Similar scores → consistent clients
- Large outlier → possible anomaly

---

## Expected Behavior

Without attack:

- FedAvg: stable
- FedProx: stable
- Krum: similar to FedAvg

With non-IID data:

- FedProx improves stability
- Krum may slightly degrade (not optimized for non-IID)

---

## Current Features

- FedAvg
- FedProx
- Krum
- Dirichlet non-IID
- Distributed evaluation
- Metric aggregation
- Plotting

---

## Planned Features

- Multi-Krum
- Trimmed Mean
- FLTrust
- FLIP
- Byzantine attack simulation
- Backdoor attack
- Robust aggregation benchmarking

---

## Notes

This project is intended for research purposes:

- Not optimized for production
- Focused on experimentation and extensibility

---

## Author

Nguyễn Thái Hoài Nam
