"""
Evaluation Pipeline
====================
Runs all four algorithms on the SAME network with the SAME traffic scenarios.
Saves results to CSV for plotting. ALL metrics are computed, NOT hardcoded.

FIXES
-----
1. Metric key mismatches fixed throughout:
   - _get_episode_metrics() returns "avg_packet_loss" and "avg_throughput"
   - After evaluate() prefixes with "avg_" they become:
       "avg_avg_packet_loss" and "avg_avg_throughput"
   - _get() fallback chains now match these exact keys for ALL algorithms.

2. MADDPG num_agents config comment cleaned up — now always uses 3 agents
   (already fixed in maddpg_agent.py to read from env, but kept explicit here).

3. CSV writer now uses a single consistent column "Throughput_Gbps" so
   plot_results.py doesn't need to guess between Mbps/Gbps variants.

4. _print_result() key chains updated to match fixed metric names.
"""

import inspect
import numpy as np
import os
import csv
import json
import time
from collections import defaultdict

from config import EVAL_CONFIG, MAPPO_CONFIG, MADDPG_CONFIG, NUM_NODES
from network_env import HealthcareSDNEnv
from mappo_agent import MAPPOAgent
from maddpg_agent import MADDPGAgent
from baselines import DijkstraRouter, DistanceVectorRouter


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def _get(res, *keys):
    """Try multiple key names in order, return first match or 0."""
    for k in keys:
        if k in res:
            return res[k]
    return 0


def _extract_metrics(res):
    """
    Return (latency, loss, throughput_gbps, jitter) from a result dict
    regardless of whether it came from a DRL agent or a baseline router.

    Key mapping:
      DRL agents  → avg_avg_latency, avg_avg_packet_loss,
                    avg_avg_throughput, avg_avg_jitter
      Baselines   → same (baselines.evaluate() uses _get_episode_metrics()
                    keys then prefixes with avg_)
    """
    lat  = _get(res,
                "avg_avg_latency",
                "avg_avg_lat")
    loss = _get(res,
                "avg_avg_packet_loss",          # DRL + baselines (fixed)
                "avg_avg_packet_loss_rate",     # legacy fallback
                "avg_packet_loss_rate")
    tput = _get(res,
                "avg_avg_throughput",           # DRL + baselines (fixed)
                "avg_avg_throughput_gbps",      # legacy fallback
                "avg_throughput_gbps")
    jit  = _get(res,
                "avg_avg_jitter",
                "avg_jitter")
    return lat, loss, tput, jit


def _print_result(res):
    lat, loss, tput, jit = _extract_metrics(res)
    print(f"  Latency: {lat:.4f}ms | Loss: {loss:.4f} | "
          f"Throughput: {tput:.6f} Gbps | Jitter: {jit:.4f}ms")


# ---------------------------------------------------------------------------
# DRL model evaluator
# ---------------------------------------------------------------------------

def evaluate_trained_model(agent, env, num_episodes=200, deterministic=True):
    """Evaluate a trained DRL model (MAPPO or MADDPG) on test episodes."""
    all_metrics          = defaultdict(list)
    per_class_latencies  = defaultdict(list)

    sa_sig = inspect.signature(agent.select_actions)
    supports_deterministic = "deterministic" in sa_sig.parameters

    for ep in range(num_episodes):
        obs, gs          = env.reset()
        episode_metrics  = defaultdict(list)

        for step in range(env.max_steps):
            if supports_deterministic:
                result = agent.select_actions(obs, deterministic=deterministic)
            else:
                result = agent.select_actions(obs)

            # MAPPO returns (actions, log_probs); MADDPG returns dict
            actions = result[0] if isinstance(result, tuple) else result

            obs, gs, rewards, done, info = env.step(actions)

            step_m = info.get("step_metrics", {})
            for key, val in step_m.items():
                if isinstance(val, (int, float)):
                    episode_metrics[key].append(val)
                elif isinstance(val, dict) and key == "per_class_latency":
                    for cls, lat in val.items():
                        per_class_latencies[cls].append(lat)

            if done:
                break

        # Primary: use episode-level metrics (consistent with baselines)
        ep_m = env._get_episode_metrics()
        for key, val in ep_m.items():
            if isinstance(val, (int, float)):
                all_metrics[key].append(val)

        # Secondary: step-level fallback
        for key, vals in episode_metrics.items():
            all_metrics[f"step_{key}"].append(np.mean(vals) if vals else 0.0)

    results = {}
    for key, vals in all_metrics.items():
        results[f"avg_{key}"] = float(np.mean(vals))
        results[f"std_{key}"] = float(np.std(vals))

    for cls in range(4):
        lats = per_class_latencies.get(cls, [0])
        results[f"class_{cls}_latency"] = float(np.mean(lats))

    return results


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def run_full_evaluation(results_dir="results", checkpoints_dir="checkpoints"):
    """
    Run complete evaluation of all four algorithms.
    All metrics are computed from actual simulation runs — nothing hardcoded.
    """
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("FULL EVALUATION PIPELINE")
    print("=" * 70)

    num_episodes = EVAL_CONFIG["num_test_episodes"]
    all_results  = {}

    # ------------------------------------------------------------------
    # 1. MAPPO
    # ------------------------------------------------------------------
    print("\n[1/4] Evaluating MAPPO...")
    env_mappo   = HealthcareSDNEnv(
        num_nodes=NUM_NODES, num_agents=MAPPO_CONFIG["num_agents"], seed=100
    )
    mappo_agent = MAPPOAgent(env_mappo, MAPPO_CONFIG)

    model_path = os.path.join(checkpoints_dir, "mappo_final.pt")
    if os.path.exists(model_path):
        mappo_agent.load(model_path)
        print(f"  Loaded trained MAPPO from {model_path}")
    else:
        print(f"  WARNING: No trained model at {model_path} — using random policy")

    t0 = time.time()
    mappo_results = evaluate_trained_model(mappo_agent, env_mappo, num_episodes)
    mappo_results["algorithm"]  = "MAPPO"
    mappo_results["eval_time"]  = time.time() - t0
    all_results["MAPPO"]        = mappo_results
    _print_result(mappo_results)

    # ------------------------------------------------------------------
    # 2. MADDPG
    # ------------------------------------------------------------------
    print("\n[2/4] Evaluating MADDPG...")
    maddpg_eval_cfg             = MADDPG_CONFIG.copy()
    maddpg_eval_cfg["num_agents"] = 3          # always 3 to cover all zones
    env_maddpg                  = HealthcareSDNEnv(
        num_nodes=NUM_NODES, num_agents=3, seed=100
    )
    maddpg_agent                = MADDPGAgent(env_maddpg, maddpg_eval_cfg)

    model_path = os.path.join(checkpoints_dir, "maddpg_final.pt")
    if os.path.exists(model_path):
        maddpg_agent.load(model_path)
        print(f"  Loaded trained MADDPG from {model_path}")
    else:
        print(f"  WARNING: No trained model at {model_path} — using random policy")

    t0 = time.time()
    maddpg_results = evaluate_trained_model(maddpg_agent, env_maddpg, num_episodes)
    maddpg_results["algorithm"] = "MADDPG"
    maddpg_results["eval_time"] = time.time() - t0
    all_results["MADDPG"]       = maddpg_results
    _print_result(maddpg_results)

    # ------------------------------------------------------------------
    # 3. Dijkstra
    # ------------------------------------------------------------------
    print("\n[3/4] Evaluating Dijkstra...")
    env_dijkstra = HealthcareSDNEnv(num_nodes=NUM_NODES, num_agents=3, seed=100)
    dijkstra     = DijkstraRouter(env_dijkstra)

    t0 = time.time()
    dijkstra_results = dijkstra.evaluate(num_episodes)
    dijkstra_results["eval_time"] = time.time() - t0
    all_results["Dijkstra"]       = dijkstra_results
    _print_result(dijkstra_results)

    # ------------------------------------------------------------------
    # 4. Distance Vector
    # ------------------------------------------------------------------
    print("\n[4/4] Evaluating Distance Vector...")
    env_dv = HealthcareSDNEnv(num_nodes=NUM_NODES, num_agents=3, seed=100)
    dv     = DistanceVectorRouter(env_dv)

    t0 = time.time()
    dv_results = dv.evaluate(num_episodes)
    dv_results["eval_time"]      = time.time() - t0
    all_results["DistanceVector"] = dv_results
    _print_result(dv_results)

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    results_file = os.path.join(results_dir, "evaluation_results.json")
    clean_results = {}
    for algo, res in all_results.items():
        clean_results[algo] = {
            k: (float(v) if isinstance(v, (np.integer, np.floating))
                else v.tolist() if isinstance(v, np.ndarray)
                else v)
            for k, v in res.items()
        }
    with open(results_file, "w") as f:
        json.dump(clean_results, f, indent=2)

    # ------------------------------------------------------------------
    # Save CSV — single consistent column set for plot_results.py
    # ------------------------------------------------------------------
    csv_file = os.path.join(results_dir, "comparison_results.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Algorithm", "Avg_Latency_ms", "Packet_Loss_Rate",
            "Throughput_Gbps",              # always Gbps — no Mbps ambiguity
            "Avg_Jitter_ms", "Eval_Time_s",
            "Critical_Latency", "High_Latency",
            "Medium_Latency",   "Normal_Latency",
        ])
        for algo, res in all_results.items():
            lat, loss, tput, jit = _extract_metrics(res)
            writer.writerow([
                algo,
                lat,
                loss,
                tput,
                jit,
                res.get("eval_time", 0),
                res.get("class_0_latency", 0),
                res.get("class_1_latency", 0),
                res.get("class_2_latency", 0),
                res.get("class_3_latency", 0),
            ])

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Results saved to:")
    print(f"  JSON : {results_file}")
    print(f"  CSV  : {csv_file}")
    print(f"{'=' * 70}")
    print(f"\n{'Algorithm':<16} {'Latency(ms)':<14} {'Loss Rate':<12} "
          f"{'Throughput(Gbps)':<18} {'Jitter(ms)':<12}")
    print("-" * 72)
    for algo, res in all_results.items():
        lat, loss, tput, jit = _extract_metrics(res)
        print(f"{algo:<16} {lat:<14.4f} {loss:<12.4f} {tput:<18.6f} {jit:<12.4f}")

    return all_results


if __name__ == "__main__":
    run_full_evaluation()