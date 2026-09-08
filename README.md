# MAPPO-based Healthcare SDN Routing

## Multi-Agent Proximal Policy Optimization for Smart Hospital Network Management

**Base Paper:** Dake et al. (2021) — "Multi-Agent Reinforcement Learning Framework in SDN-IoT for Transient Load Detection and Prevention"

**Improvement:** MAPPO algorithm with healthcare-specific traffic prioritization on a 30-node hospital SDN network.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Quick Test (5 minutes)
```bash
python train_and_evaluate.py --quick
```

### 3. Full Training (2-4 hours GPU, 8-12 hours CPU)
```bash
python train_and_evaluate.py
```

---

## Project Structure

```
smart_healthcare_routing/
├── config.py                    # Hyperparameters and network configuration
├── network_env.py               # Healthcare SDN simulation environment
├── mappo_agent.py               # MAPPO implementation
├── maddpg_agent.py              # MADDPG baseline implementation
├── baselines.py                 # Dijkstra and Distance Vector baselines
├── evaluate_all.py              # Evaluate all routing algorithms
├── plot_results.py              # Generate comparison visualizations
├── train_and_evaluate.py        # Main training and evaluation script
├── training_notebook.ipynb      # Jupyter notebook for experimentation
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── project_images/              # Generated plots and visualizations
```

---

## Algorithms Compared

| Algorithm | Type | Key Feature |
|-----------|------|-------------|
| **MAPPO** (Ours) | Multi-Agent DRL, On-Policy | Healthcare priority-aware, shared critic, PPO clipping |
| **MADDPG** (Base Paper) | Multi-Agent DRL, Off-Policy | Replay buffer, separate critics, DDPG-based |
| **Dijkstra** | Traditional | Static shortest path, no adaptation |
| **Distance Vector** | Traditional | Hop-count based, slow convergence |

---

## Healthcare Traffic Classes

| Priority | Class | Examples | Max Latency | Reward Weight |
|----------|-------|----------|-------------|---------------|
| CRITICAL | 0 | ICU alerts, surgical monitors | 5ms | 4.0x |
| HIGH | 1 | Patient vitals, bed monitors | 20ms | 2.0x |
| MEDIUM | 2 | CT/MRI imaging, lab results | 100ms | 1.0x |
| NORMAL | 3 | Admin email, billing, EHR | 500ms | 0.5x |

---

## Key Improvements Over Base Paper

1. **Algorithm:** MADDPG → MAPPO (on-policy, more stable, PPO clipping)
2. **Domain:** Generic IoT → Healthcare-specific with 4 traffic priority classes
3. **Scale:** 5-8 switches → 30-node hospital network topology
4. **Comparison:** DDPG only → MAPPO vs MADDPG vs Dijkstra vs Distance Vector
5. **Reward:** Simple 1/U → Priority-weighted latency + packet loss + jitter + throughput

---

## No Hardcoded Results

Every metric, every data point on every graph is computed from actual simulation runs.
The environment computes metrics using queuing theory (M/M/1 model).
Training logs are saved to CSV. Evaluation results are saved to JSON/CSV.
Plots read from these files. Nothing is fabricated.
