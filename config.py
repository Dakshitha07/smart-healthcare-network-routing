"""
Configuration for MAPPO-based Healthcare SDN Routing
=====================================================
All hyperparameters, network parameters, and healthcare traffic profiles.
Based on improvement over Dake et al. (2021) MADDPG SDN-IoT framework.

FIXES APPLIED
-------------
REWARD_CONFIG was the root cause of always-negative rewards:

1. latency_penalty_scale: 1.0 → 0.1
   With avg_latency ~50ms, penalty was -50 per step before scaling.
   Reducing to 0.1 makes it -5, which the throughput bonus can offset.

2. packet_loss_penalty: 10.0 → 5.0
   A 30% loss rate was giving -3.0 penalty alone. Halved so good routing
   (low loss) doesn't still dominate the reward negatively.

3. throughput_reward: 0.5 → 50.0
   Throughput is measured in Gbps — typically ~0.001 Gbps per step.
   At 0.5 scale the bonus was ~0.0005, essentially zero. At 50.0 it
   becomes ~0.05, a meaningful positive contribution.

4. jitter_penalty: 1.0 → 0.1
   Jitter values are in ms and can be large. Scaled down to match latency.

5. priority_latency_penalty weights: unchanged in value but the latency
   values fed into them are now smaller due to fix 1, so they no longer
   dominate the reward signal.

6. reward_scaling: 0.01 → 0.1
   After balancing the components above, scaling by 0.01 made all rewards
   tiny (-0.5 to +0.5). Increasing to 0.1 gives a clearer learning signal
   in the range roughly -5 to +5 for a trained agent.

7. delivery_bonus: 0.5 added per delivered packet — gives the agent a
   direct positive signal for successfully routing flows, which was
   completely absent before (only penalties existed).
"""

import numpy as np

# ==============================================================================
# TRAINING CONFIGURATION (MAPPO)
# ==============================================================================
MAPPO_CONFIG = {
    # Learning rates
    "lr_actor":          1e-4,
    "lr_critic":         3e-4,
    "num_envs":          15,
    "max_steps":         256,
    "ppo_epochs":        10,
    "clip_coeff":        0.2,
    "entropy_coeff":     0.01,
    "value_loss_coeff":  0.5,
    "max_grad_norm":     0.5,
    "gae_lambda":        0.95,
    "gamma":             0.99,
    "total_timesteps":   2_000_000,
    "batch_size":        256,
    "minibatch_size":    64,
    "num_agents":        3,
    "hidden_dim":        256,
    "num_layers":        3,
    "activation":        "relu",
    "log_interval":      10,
    "save_interval":     100,
    "eval_interval":     200,
    "seed":              42,
    "anneal_lr":         True,
}

# ==============================================================================
# TRAINING CONFIGURATION (MADDPG - Baseline)
# ==============================================================================
MADDPG_CONFIG = {
    "lr_actor":              1e-4,
    "lr_critic":             1e-3,
    "num_agents":            3,
    "hidden_dim":            128,
    "gamma":                 0.99,
    "tau":                   0.005,
    "batch_size":            64,
    "buffer_size":           100_000,
    "epsilon_start":         1.0,
    "epsilon_end":           0.05,
    "epsilon_decay":         0.995,
    "total_episodes":        3000,
    "max_steps_per_episode": 256,
    "seed":                  42,
}

# ==============================================================================
# NETWORK TOPOLOGY CONFIGURATION
# ==============================================================================
NUM_NODES  = 30
NUM_AGENTS = MAPPO_CONFIG["num_agents"]

HOSPITAL_ZONES = {
    "critical_care": {
        "nodes": list(range(0, 8)),
        "departments": {
            "ICU": [0, 1, 2],
            "ER":  [3, 4, 5],
            "OR":  [6, 7],
        },
        "priority_class":  0,
        "max_latency_ms":  5.0,
        "color": "#E24B4A",
    },
    "monitoring": {
        "nodes": list(range(8, 16)),
        "departments": {
            "General_Ward_A": [8,  9],
            "General_Ward_B": [10, 11],
            "Pediatrics":     [12, 13],
            "Maternity":      [14, 15],
        },
        "priority_class":  1,
        "max_latency_ms":  20.0,
        "color": "#EF9F27",
    },
    "diagnostic": {
        "nodes": list(range(16, 22)),
        "departments": {
            "Radiology":  [16, 17, 18],
            "Laboratory": [19, 20, 21],
        },
        "priority_class":  2,
        "max_latency_ms":  100.0,
        "color": "#378ADD",
    },
    "administrative": {
        "nodes": list(range(22, 27)),
        "departments": {
            "Pharmacy":     [22, 23],
            "Admin_Office": [24, 25, 26],
        },
        "priority_class":  3,
        "max_latency_ms":  500.0,
        "color": "#888780",
    },
    "infrastructure": {
        "nodes": list(range(27, 30)),
        "departments": {
            "Data_Center":    [27],
            "SDN_Controller": [28],
            "Gateway":        [29],
        },
        "priority_class":  1,
        "max_latency_ms":  10.0,
        "color": "#7F77DD",
    },
}

TRAFFIC_CLASSES = {
    0: {"name": "critical", "weight": 4.0, "label": "Emergency/ICU Alerts"},
    1: {"name": "high",     "weight": 2.0, "label": "Patient Monitoring"},
    2: {"name": "medium",   "weight": 1.0, "label": "Medical Imaging"},
    3: {"name": "normal",   "weight": 0.5, "label": "Admin/EHR Traffic"},
}

TRAFFIC_PROFILES = {
    0: {  # CRITICAL
        "data_rate_range":   (1, 10),
        "packet_size_range": (64, 256),
        "arrival_rate":      0.8,
        "burst_probability": 0.15,
    },
    1: {  # HIGH
        "data_rate_range":   (10, 100),
        "packet_size_range": (128, 512),
        "arrival_rate":      0.6,
        "burst_probability": 0.10,
    },
    2: {  # MEDIUM
        "data_rate_range":   (10000, 500000),
        "packet_size_range": (1024, 65536),
        "arrival_rate":      0.3,
        "burst_probability": 0.05,
    },
    3: {  # NORMAL
        "data_rate_range":   (500, 5000),
        "packet_size_range": (512, 1500),
        "arrival_rate":      0.4,
        "burst_probability": 0.02,
    },
}

LINK_CONFIG = {
    "intra_zone_bandwidth":  1000,    # Mbps
    "inter_zone_bandwidth":  10000,   # Mbps
    "backbone_bandwidth":    40000,   # Mbps
    "intra_zone_delay":      0.1,     # ms
    "inter_zone_delay":      0.5,     # ms
    "backbone_delay":        0.05,    # ms
    "buffer_size":           500,     # packets
    "processing_delay":      0.01,    # ms
}

# ==============================================================================
# REWARD CONFIGURATION  ← THIS WAS THE ROOT CAUSE
# ==============================================================================
REWARD_CONFIG = {
    # Latency penalty: increased from 0.1 → 0.3 so latency reduction
    # produces a meaningful reward improvement the agent can learn from
    "latency_penalty_scale": 0.3,

    # Reduced further from 5.0 → 2.0: 91% loss was still giving -1.82
    # penalty per step, drowning out the delivery bonus completely
    "packet_loss_penalty": 2.0,

    # Jitter penalty kept small
    "jitter_penalty": 0.1,

    # Throughput bonus: Gbps * 50 gives ~0.05 per step
    "throughput_reward": 50.0,

    # Increased from 0.5 → 1.0: each delivered packet now gives a
    # meaningful positive signal that the agent can learn from
    "delivery_bonus": 1.0,

    "priority_weights": {
        0: 4.0,   # Critical
        1: 2.0,   # High
        2: 1.0,   # Medium
        3: 0.5,   # Normal
    },

    # reward_scaling: 0.1 keeps rewards in roughly [-5, +5] range
    "reward_scaling": 0.1,
}

# ==============================================================================
# EVALUATION CONFIGURATION
# ==============================================================================
EVAL_CONFIG = {
    "num_test_episodes": 200,
    "traffic_scenarios": ["low", "medium", "high", "burst", "mixed"],
    "traffic_load_multipliers": {
        "low":    0.3,
        "medium": 0.6,
        "high":   0.9,
        "burst":  1.5,
        "mixed":  0.7,
    },
    "algorithms": ["MAPPO", "MADDPG", "Dijkstra", "DistanceVector"],
}

# ==============================================================================
# EXPERIMENT VARIATIONS
# ==============================================================================
VARIATION_CONFIGS = {
    "learning_rates":  [1e-4, 2.5e-4, 5e-4, 1e-3],
    "network_sizes":   [30, 45, 60],
    "entropy_coeffs":  [0.01, 0.02, 0.05],
}

# ==============================================================================
# PATHS
# ==============================================================================
PATHS = {
    "results_dir":    "results",
    "checkpoints_dir": "checkpoints",
    "images_dir":     "project_images",
    "webapp_dir":     "webapp",
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_node_zone(node_id):
    """Return the zone name for a given node ID."""
    for zone_name, zone_info in HOSPITAL_ZONES.items():
        if node_id in zone_info["nodes"]:
            return zone_name
    return "unknown"

def get_node_priority(node_id):
    """Return the traffic priority class for a given node ID."""
    for zone_name, zone_info in HOSPITAL_ZONES.items():
        if node_id in zone_info["nodes"]:
            return zone_info["priority_class"]
    return 3

def get_node_department(node_id):
    """Return the department name for a given node ID."""
    for zone_name, zone_info in HOSPITAL_ZONES.items():
        for dept_name, dept_nodes in zone_info["departments"].items():
            if node_id in dept_nodes:
                return dept_name
    return "Unknown"

def get_priority_weight(traffic_class):
    """Return the reward weight for a traffic class."""
    return REWARD_CONFIG["priority_weights"].get(traffic_class, 0.5)