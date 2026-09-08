# MAPPO-based Healthcare SDN Routing

## Multi-Agent Proximal Policy Optimization for Smart Hospital Network Management

A multi-agent deep reinforcement learning framework for intelligent and
priority-aware routing in smart healthcare Software-Defined Networks (SDN).

The project implements **Multi-Agent Proximal Policy Optimization (MAPPO)**
using **Centralized Training with Decentralized Execution (CTDE)** for dynamic
routing in a 30-node hospital network.

---

## Project Overview

Healthcare networks carry different types of traffic with different levels
of urgency. Critical traffic such as ICU alerts and surgical monitoring
requires significantly lower latency than administrative traffic.

This project develops a healthcare-specific SDN routing environment where
traffic is divided into four priority classes. MAPPO agents learn dynamic
routing policies while considering network conditions and healthcare traffic
priorities.

The proposed approach is evaluated against:

- MAPPO
- MADDPG
- Dijkstra
- Distance Vector

---


## Base Paper

**Dake et al. (2021)**

*"Multi-Agent Reinforcement Learning Framework in SDN-IoT for Transient Load
Detection and Prevention"*

### Proposed Improvement

The original approach is extended by replacing the MADDPG-based approach with
MAPPO and introducing healthcare-specific traffic prioritisation on a
30-node hospital SDN network.

---

## Objectives

- Develop a Centralized Training with Decentralized Execution (CTDE) based
  MAPPO routing framework.
- Implement healthcare-specific traffic prioritisation.
- Enable dynamic routing using multi-agent reinforcement learning.
- Evaluate MAPPO against MADDPG and traditional routing algorithms.
- Analyse network performance using latency, packet loss, throughput, and
  jitter.

---

## System Architecture

The system consists of:

1. A 30-node hospital SDN network environment.
2. Multiple MAPPO agents responsible for routing decisions.
3. Healthcare-specific traffic priority classes.
4. A centralized critic during training.
5. Decentralized policies during execution.
6. Baseline routing algorithms for comparison.
7. Evaluation and visualization pipelines.

---

## Healthcare Traffic Classes

| Priority | Class | Examples | Maximum Latency | Reward Weight |
|----------|-------|----------|-----------------|---------------|
| CRITICAL | 0 | ICU alerts, surgical monitors | 5 ms | 4.0x |
| HIGH | 1 | Patient vitals, bed monitors | 20 ms | 2.0x |
| MEDIUM | 2 | CT/MRI imaging, lab results | 100 ms | 1.0x |
| NORMAL | 3 | Administrative email, billing, EHR | 500 ms | 0.5x |

The routing environment gives higher importance to critical healthcare
traffic while continuing to manage lower-priority traffic.

---

## Algorithms Compared

| Algorithm | Type | Key Feature |
|-----------|------|-------------|
| **MAPPO (Proposed)** | Multi-Agent DRL, On-Policy | Healthcare priority-aware routing, shared critic, PPO clipping |
| **MADDPG** | Multi-Agent DRL, Off-Policy | Replay buffer, separate critics, DDPG-based learning |
| **Dijkstra** | Traditional | Static shortest-path routing |
| **Distance Vector** | Traditional | Hop-count based routing |

---

## Key Improvements Over the Base Paper

### 1. Algorithm

MADDPG → **MAPPO**

MAPPO uses on-policy learning and PPO clipping to provide more stable policy
updates.

### 2. Application Domain

Generic IoT → **Healthcare-specific networking**

The environment introduces four healthcare traffic priority classes.

### 3. Network Scale

Small SDN topology → **30-node hospital network**

### 4. Comparative Evaluation

The project compares:

**MAPPO vs MADDPG vs Dijkstra vs Distance Vector**

### 5. Reward Design

The routing objective considers multiple Quality of Service (QoS) metrics:

- Priority-weighted latency
- Packet loss
- Jitter
- Throughput

---

## Results

The four routing approaches were evaluated using the same healthcare SDN
environment and QoS metrics.

| Algorithm | Latency (ms) | Packet Loss | Throughput (Gbps) | Jitter (ms) |
|-----------|-------------:|------------:|------------------:|------------:|
| **MAPPO** | **1.2285** | **0.0027** | **0.000506** | **0.2503** |
| MADDPG | 1.2617 | 0.0057 | 0.000501 | 0.2829 |
| Dijkstra | 1.7254 | 0.2939 | 0.000458 | 0.3516 |
| Distance Vector | 1.7037 | 0.2543 | 0.000464 | 0.3436 |

**MAPPO achieved the best overall performance across the evaluated QoS
metrics.**

The results indicate that the proposed approach can reduce latency, packet
loss, and jitter while improving throughput compared with the evaluated
baselines.

---

## Training Analysis

MAPPO demonstrated more stable training behaviour than MADDPG.

The project analysis attributes this stability to PPO clipping, which helps
avoid unstable policy updates during training.

MAPPO also learned to prioritise healthcare traffic, routing critical traffic
with lower latency while managing lower-priority traffic separately.

---

## Project Visualizations

The repository includes generated visualizations covering:

- MAPPO reward convergence
- MAPPO loss curves
- MAPPO latency during training
- MAPPO entropy
- MADDPG reward curve
- MADDPG latency during training
- MADDPG entropy
- MAPPO vs MADDPG training comparison
- Latency comparison
- Packet loss comparison
- Throughput comparison
- Jitter comparison
- Per-class latency comparison
- Performance radar chart
- 30-node hospital network topology

All visualizations are available in:

`project_images/`

---

## Project Structure

```text
smart-healthcare-network-routing/
│
├── config.py
│   └── Hyperparameters and network configuration
│
├── network_env.py
│   └── Healthcare SDN simulation environment
│
├── mappo_agent.py
│   └── MAPPO implementation
│
├── maddpg_agent.py
│   └── MADDPG baseline implementation
│
├── baselines.py
│   └── Dijkstra and Distance Vector implementations
│
├── evaluate_all.py
│   └── Evaluate all routing algorithms
│
├── plot_results.py
│   └── Generate training and comparison visualizations
│
├── train_and_evaluate.py
│   └── Main training and evaluation script
│
├── training_notebook.ipynb
│   └── Jupyter notebook for experimentation
│
├── requirements.txt
│   └── Python dependencies
│
├── README.md
│   └── Project documentation
│
├── .gitignore
│   └── Git ignored files
│
└── project_images/
    └── Generated plots and network visualizations
~~~
```
## Technologies Used

- **Python**
- **PyTorch**
- **NumPy**
- **NetworkX**
- **Pandas**
- **Matplotlib**
- **SciPy**
- **Jupyter Notebook**

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Dakshitha07/smart-healthcare-network-routing.git
cd smart-healthcare-network-routing

```
### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Quick Test

A smaller configuration can be used to test the training and evaluation pipeline.

```bash
python train_and_evaluate.py --quick
```

### Full Training

Run the complete training pipeline using:

```bash
python train_and_evaluate.py
```

The pipeline trains the reinforcement learning models, evaluates the routing algorithms, and generates the required visualizations.

---

## Evaluation

The evaluation pipeline compares four routing algorithms under the simulated healthcare network environment:

- **MAPPO**
- **MADDPG**
- **Dijkstra**
- **Distance Vector**

The following Quality of Service (QoS) metrics are evaluated:

- Average latency
- Packet loss
- Throughput
- Average jitter
- Per-class latency

This evaluation allows the proposed MAPPO approach to be compared against both reinforcement learning and traditional routing methods.

---

## Key Findings

- **MAPPO achieved the lowest average latency** among the evaluated approaches.
- **MAPPO achieved the lowest packet loss.**
- **MAPPO achieved the highest throughput.**
- **MAPPO achieved the lowest jitter.**
- MAPPO demonstrated more stable training behaviour than MADDPG.
- Healthcare traffic prioritisation gives greater importance to critical traffic.
- Overall, MAPPO provided the strongest QoS performance in the simulated healthcare network.

---

## Future Improvements

Possible future extensions include:

- Testing larger hospital network topologies.
- Evaluating additional multi-agent reinforcement learning algorithms.
- Testing the framework with real-world network traces.
- Introducing additional healthcare traffic types.
- Evaluating performance under different network failure conditions.
- Investigating real SDN controller integration.
- Evaluating the approach on larger and more dynamic network environments.

---

## Conclusion

This project presents a **MAPPO-based multi-agent deep reinforcement learning framework for smart healthcare SDN routing**.

The framework combines:

- Multi-agent reinforcement learning
- Centralized Training with Decentralized Execution
- Healthcare traffic prioritisation
- Dynamic routing
- Multi-metric QoS evaluation

The proposed MAPPO approach was evaluated against **MADDPG, Dijkstra, and Distance Vector** routing algorithms.

Based on the evaluated simulation results, **MAPPO achieved the strongest overall performance across latency, packet loss, throughput, and jitter**.

The project demonstrates the potential of multi-agent deep reinforcement learning for intelligent and priority-aware routing in smart healthcare network environments.
