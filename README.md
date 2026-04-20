# Federated Learning Baseline with Flower + PyTorch

## Overview

This project implements a baseline Federated Learning (FL) system using
Flower and PyTorch for distributed training on CIFAR-10.

Currently supported aggregation strategies:

- FedAvg
- FedProx

This project is designed as a baseline framework for future research on:

- Byzantine-robust aggregation
- Backdoor defenses
- Robust federated optimization
- Custom aggregation algorithms (Krum, Multi-Krum, FLTrust, FLIP, etc.)

---

## What is Federated Learning?

Federated Learning (FL) is a distributed machine learning setting where:

- Multiple clients train locally on private data
- Clients do not send raw data to the server
- Clients only send model updates (weights/gradients)
- A central server aggregates those updates into a global model

Typical FL workflow:

1. Server initializes global model
2. Server sends model to clients
3. Clients train locally
4. Clients send updates to server
5. Server aggregates updates
6. Repeat for multiple rounds

---

## Problem Being Studied

This project studies:

- Distributed training under the Federated Learning setting
- Effect of aggregation strategies on convergence
- Comparison between:

### FedAvg

Standard Federated Averaging algorithm.

Global update:

w(t+1) = Σ(n_k / N) \* w_k

---

### FedProx

FedAvg with proximal regularization to improve training under heterogeneous data.

Objective:

F_k(w) + (μ/2)||w - w_global||²

---

## Dataset

Dataset used:

- CIFAR-10

10 image classes:

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

Current data partition:

- IID random split across clients

Future work:

- Non-IID Dirichlet split
- Label-skew partition

---

## Model

Current model:

Simple CNN:

- Conv2d(3,32)
- Conv2d(32,64)
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
│   └── strategy_factory.py

├── aggregators/
│   └── metrics.py

├── data/
│   └── CIFAR10 dataset

└── fl_env/
```

---

## Installation

Create environment:

```bash
python -m venv fl_env
```

Activate:

Windows:

```bash
fl_env\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

## Run Server (FedAvg)

```bash
python server.py --strategy fedavg
```

---

## Run Server (FedProx)

```bash
python server.py --strategy fedprox
```

---

## Custom rounds

```bash
python server.py --strategy fedavg --rounds 10
```

---

## Start Clients

Terminal 1:

```bash
python client.py --partition-id 0 --num-clients 2
```

Terminal 2:

```bash
python client.py --partition-id 1 --num-clients 2
```

---

## Example with 2 clients

Start:

1. Run server

```bash
python server.py --strategy fedavg
```

2. Run Client 0

```bash
python client.py --partition-id 0 --num-clients 2
```

3. Run Client 1

```bash
python client.py --partition-id 1 --num-clients 2
```

Training will run for 5 rounds.

Results saved as:

```text
results_fedavg.png
```

---

## Current Features

Implemented:

- FedAvg
- FedProx
- Weighted metric aggregation
- Distributed evaluation
- Accuracy/Loss plotting

---

## Planned Features

Planned:

- Krum
- Multi-Krum
- Trimmed Mean
- FLTrust
- FLIP

- Byzantine clients
- Backdoor attack simulation
- Non-IID partitioning

---

## Notes

This repository is intended as a research baseline, not a production FL system.

Current implementation focuses on:

- Simplicity
- Extensibility
- Aggregation research

---

## Author

Federated Learning Research Baseline
