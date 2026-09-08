"""
Plot Results
=============
Generates all comparison and training graphs from REAL computed data.
Reads from CSV/JSON files produced by training and evaluation.
NO hardcoded values — every point on every graph comes from actual results.

FIXES
-----
1. Throughput column: evaluate_all.py now always writes "Throughput_Gbps".
   plot_results.py previously tried "Throughput_Mbps" first, causing
   scale confusion (Gbps values displayed as if they were Mbps).
   Now reads "Throughput_Gbps" directly and converts to Mbps for display.

2. comparison_results.csv no longer has a "Throughput_Mbps" column at all,
   so the old col = "Throughput_Mbps" if ... else "Throughput_Gbps" guard
   has been replaced with a direct read + ×1000 conversion for display.

3. plot_network_topology() moved the HealthcareSDNEnv import inside the
   function to avoid a circular import if plot_results is imported early.

4. plot_multi_seed_summary() now handles the Throughput_Gbps→Mbps scale
   conversion consistently with the single-seed charts.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os

plt.rcParams.update({
    "font.size":        12,
    "axes.titlesize":   14,
    "axes.labelsize":   12,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.figsize":   (10, 6),
    "figure.dpi":       150,
    "savefig.dpi":      150,
    "savefig.bbox":     "tight",
})

COLORS = {
    "MAPPO":          "#534AB7",
    "MADDPG":         "#1D9E75",
    "Dijkstra":       "#D85A30",
    "DistanceVector": "#888780",
}


def smooth(data, window=20):
    if len(data) < window:
        return data
    return pd.Series(data).rolling(window=window, min_periods=1).mean().values


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(results_dir="results", output_dir="project_images"):
    os.makedirs(output_dir, exist_ok=True)

    # ---- MAPPO ----
    mappo_log = os.path.join(results_dir, "mappo_training_log.csv")
    if os.path.exists(mappo_log):
        df = pd.read_csv(mappo_log)

        # 1. Reward
        fig, ax = plt.subplots(figsize=(10, 5))
        sm = smooth(df["reward"], 30)
        ax.plot(df["episode"], sm, color=COLORS["MAPPO"],
                linewidth=1.5, label="MAPPO Reward (smoothed)")
        std = df["reward"].rolling(30).std().fillna(0)
        ax.fill_between(df["episode"], sm - std, sm + std,
                        alpha=0.15, color=COLORS["MAPPO"])
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Reward")
        ax.set_title("MAPPO Training — Reward Convergence")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "mappo_reward_curve.png"))
        plt.close()
        print("  Saved: mappo_reward_curve.png")

        # 2. Loss curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(df["episode"], smooth(df["actor_loss"],  30),
                 color="#E24B4A", linewidth=1.2)
        ax1.set_xlabel("Episode"); ax1.set_ylabel("Actor Loss")
        ax1.set_title("MAPPO Actor Loss"); ax1.grid(True, alpha=0.3)

        ax2.plot(df["episode"], smooth(df["critic_loss"], 30),
                 color="#378ADD", linewidth=1.2)
        ax2.set_xlabel("Episode"); ax2.set_ylabel("Critic Loss")
        ax2.set_title("MAPPO Critic Loss"); ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "mappo_loss_curves.png"))
        plt.close()
        print("  Saved: mappo_loss_curves.png")

        # 3. Latency
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df["episode"], smooth(df["avg_latency"], 30),
                color=COLORS["MAPPO"], linewidth=1.5, label="Avg Latency")
        if "critical_latency" in df.columns:
            ax.plot(df["episode"], smooth(df["critical_latency"], 30),
                    color="#E24B4A", linewidth=1, linestyle="--",
                    label="Critical Class")
            ax.plot(df["episode"], smooth(df["normal_latency"],   30),
                    color="#888780", linewidth=1, linestyle="--",
                    label="Normal Class")
        ax.set_xlabel("Episode"); ax.set_ylabel("Latency (ms)")
        ax.set_title("MAPPO Training — Latency over Episodes")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "mappo_latency_training.png"))
        plt.close()
        print("  Saved: mappo_latency_training.png")

        # 4. Entropy
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df["episode"], smooth(df["entropy"], 30),
                color="#1D9E75", linewidth=1.2)
        ax.set_xlabel("Episode"); ax.set_ylabel("Entropy")
        ax.set_title("MAPPO Policy Entropy (Exploration → Exploitation)")
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "mappo_entropy.png"))
        plt.close()
        print("  Saved: mappo_entropy.png")
    else:
        print(f"  WARNING: {mappo_log} not found. Train MAPPO first.")

    # ---- MADDPG ----
    maddpg_log = os.path.join(results_dir, "maddpg_training_log.csv")
    if os.path.exists(maddpg_log):
        df = pd.read_csv(maddpg_log)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df["episode"], smooth(df["reward"], 30),
                color=COLORS["MADDPG"], linewidth=1.5, label="MADDPG Reward")
        ax.set_xlabel("Episode"); ax.set_ylabel("Cumulative Reward")
        ax.set_title("MADDPG Training — Reward Convergence (Baseline)")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "maddpg_reward_curve.png"))
        plt.close()
        print("  Saved: maddpg_reward_curve.png")

        # MADDPG Latency training curve
        if "avg_latency" in df.columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df["episode"], smooth(df["avg_latency"], 30),
                    color=COLORS["MADDPG"], linewidth=1.5, label="Avg Latency")
            ax.set_xlabel("Episode"); ax.set_ylabel("Latency (ms)")
            ax.set_title("MADDPG Training — Latency over Episodes")
            ax.legend(); ax.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, "maddpg_latency_training.png"))
            plt.close()
            print("  Saved: maddpg_latency_training.png")

        # MADDPG Entropy training curve
        if "entropy" in df.columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df["episode"], smooth(df["entropy"], 30),
                    color=COLORS["MADDPG"], linewidth=1.2)
            ax.set_xlabel("Episode"); ax.set_ylabel("Entropy")
            ax.set_title("MADDPG Policy Entropy")
            ax.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, "maddpg_entropy.png"))
            plt.close()
            print("  Saved: maddpg_entropy.png")

    # ---- Combined ----
    if os.path.exists(mappo_log) and os.path.exists(maddpg_log):
        df_m = pd.read_csv(mappo_log)
        df_d = pd.read_csv(maddpg_log)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_m["episode"], smooth(df_m["reward"], 30),
                color=COLORS["MAPPO"],  linewidth=1.5, label="MAPPO")
        ax.plot(df_d["episode"], smooth(df_d["reward"], 30),
                color=COLORS["MADDPG"], linewidth=1.5, label="MADDPG")
        ax.set_xlabel("Episode"); ax.set_ylabel("Cumulative Reward")
        ax.set_title("Training Comparison — MAPPO vs MADDPG")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "training_comparison.png"))
        plt.close()
        print("  Saved: training_comparison.png")


# ---------------------------------------------------------------------------
# Comparison bar charts
# ---------------------------------------------------------------------------

def plot_comparison_charts(results_dir="results", output_dir="project_images"):
    os.makedirs(output_dir, exist_ok=True)

    csv_file = os.path.join(results_dir, "comparison_results.csv")
    if not os.path.exists(csv_file):
        print(f"  WARNING: {csv_file} not found. Run evaluate_all.py first.")
        return

    df         = pd.read_csv(csv_file)
    algorithms = df["Algorithm"].tolist()
    colors     = [COLORS.get(a, "#888780") for a in algorithms]

    # 1. Latency
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(algorithms, df["Avg_Latency_ms"], color=colors,
                  edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Average Latency (ms)")
    ax.set_title("Average Latency Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, df["Avg_Latency_ms"]):
        ax.text(bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + max(df["Avg_Latency_ms"]) * 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    plt.savefig(os.path.join(output_dir, "comparison_latency.png"))
    plt.close()
    print("  Saved: comparison_latency.png")

    # 2. Packet Loss
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(algorithms, df["Packet_Loss_Rate"] * 100,
                  color=colors, edgecolor="white")
    ax.set_ylabel("Packet Loss Rate (%)")
    ax.set_title("Packet Loss Rate Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, df["Packet_Loss_Rate"] * 100):
        ax.text(bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + 0.2,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
    plt.savefig(os.path.join(output_dir, "comparison_packet_loss.png"))
    plt.close()
    print("  Saved: comparison_packet_loss.png")

    # 3. Throughput — FIX: always read Throughput_Gbps, convert to Mbps for display
    fig, ax = plt.subplots(figsize=(8, 5))
    vals_mbps = df["Throughput_Gbps"] * 1000   # Gbps → Mbps
    bars = ax.bar(algorithms, vals_mbps, color=colors, edgecolor="white")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title("Throughput Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, vals_mbps):
        ax.text(bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + max(vals_mbps) * 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    plt.savefig(os.path.join(output_dir, "comparison_throughput.png"))
    plt.close()
    print("  Saved: comparison_throughput.png")

    # 4. Jitter
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(algorithms, df["Avg_Jitter_ms"], color=colors, edgecolor="white")
    ax.set_ylabel("Average Jitter (ms)")
    ax.set_title("Jitter Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, df["Avg_Jitter_ms"]):
        ax.text(bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + max(df["Avg_Jitter_ms"]) * 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    plt.savefig(os.path.join(output_dir, "comparison_jitter.png"))
    plt.close()
    print("  Saved: comparison_jitter.png")

    # 5. Per-class latency
    cls_cols = ["Critical_Latency", "High_Latency",
                "Medium_Latency",   "Normal_Latency"]
    if all(c in df.columns for c in cls_cols):
        fig, ax = plt.subplots(figsize=(12, 6))
        x      = np.arange(len(algorithms))
        width  = 0.2
        cls_colors = ["#E24B4A", "#EF9F27", "#378ADD", "#888780"]
        cls_names  = ["Critical", "High", "Medium", "Normal"]
        for i, (col, name, clr) in enumerate(zip(cls_cols, cls_names, cls_colors)):
            ax.bar(x + i * width, df[col], width,
                   label=name, color=clr, edgecolor="white")
        ax.set_xlabel("Algorithm"); ax.set_ylabel("Latency (ms)")
        ax.set_title("Per-Traffic-Class Latency — Healthcare Priority Impact")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(algorithms)
        ax.legend(title="Traffic Class")
        ax.grid(True, axis="y", alpha=0.3)
        plt.savefig(os.path.join(output_dir, "comparison_per_class_latency.png"))
        plt.close()
        print("  Saved: comparison_per_class_latency.png")

    # 6. Radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    metrics       = ["Avg_Latency_ms", "Packet_Loss_Rate", "Avg_Jitter_ms"]
    metric_labels = ["Latency\n(lower=better)",
                     "Packet Loss\n(lower=better)",
                     "Jitter\n(lower=better)"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    for algo in algorithms:
        row    = df[df["Algorithm"] == algo]
        values = []
        for m in metrics:
            max_val = df[m].max()
            values.append(
                1 - float(row[m].values[0]) / max_val if max_val > 0 else 1
            )
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=1.5,
                label=algo, color=COLORS.get(algo, "#888"))
        ax.fill(angles, values, alpha=0.1, color=COLORS.get(algo, "#888"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_title("Performance Overview\n(outer = better)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.savefig(os.path.join(output_dir, "comparison_radar.png"))
    plt.close()
    print("  Saved: comparison_radar.png")


# ---------------------------------------------------------------------------
# Network topology
# ---------------------------------------------------------------------------

def plot_network_topology(results_dir="results", output_dir="project_images"):
    os.makedirs(output_dir, exist_ok=True)

    # Import here to avoid circular import issues
    import networkx as nx
    from network_env import HealthcareSDNEnv
    from config import HOSPITAL_ZONES

    env = HealthcareSDNEnv(num_nodes=30, num_agents=3, seed=42)
    G   = env.graph

    fig, ax = plt.subplots(figsize=(14, 10))

    zone_positions = {
        "critical_care":    (0.2, 0.8),
        "monitoring":       (0.8, 0.8),
        "diagnostic":       (0.2, 0.3),
        "administrative":   (0.8, 0.3),
        "infrastructure":   (0.5, 0.1),
    }

    pos = {}
    for zone_name, zone_info in HOSPITAL_ZONES.items():
        cx, cy = zone_positions[zone_name]
        nodes  = zone_info["nodes"]
        n      = len(nodes)
        for i, node in enumerate(nodes):
            angle    = 2 * np.pi * i / max(n, 1)
            r        = 0.08 + 0.02 * (n > 4)
            pos[node] = (cx + r * np.cos(angle), cy + r * np.sin(angle))

    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5, ax=ax)

    for zone_name, zone_info in HOSPITAL_ZONES.items():
        nx.draw_networkx_nodes(G, pos,
                               nodelist=zone_info["nodes"],
                               node_color=zone_info["color"],
                               node_size=200, alpha=0.8, ax=ax)

    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)

    for zone_name, (cx, cy) in zone_positions.items():
        ax.text(cx, cy + 0.14,
                zone_name.replace("_", " ").title(),
                ha="center", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor="gray", alpha=0.8))

    ax.set_title("Hospital SDN Network Topology (30 Nodes)",
                 fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.savefig(os.path.join(output_dir, "network_topology.png"))
    plt.close()
    print("  Saved: network_topology.png")


# ---------------------------------------------------------------------------
# Multi-seed summary
# ---------------------------------------------------------------------------

def plot_multi_seed_summary(results_dir="results", output_dir="project_images",
                             seeds=(42, 123, 456)):
    seed_dfs = []
    for seed in seeds:
        path = os.path.join(results_dir, f"seed_{seed}", "comparison_results.csv")
        if os.path.exists(path):
            df        = pd.read_csv(path)
            df["seed"] = seed
            seed_dfs.append(df)

    if len(seed_dfs) < 2:
        print("  INFO: Multi-seed plots need results/seed_42/, seed_123/, seed_456/")
        print("  Run: python train_and_evaluate.py --full --seed 42  (then 123, 456)")
        return

    combined   = pd.concat(seed_dfs)
    algorithms = combined["Algorithm"].unique()
    colors     = [COLORS.get(a, "#888780") for a in algorithms]

    # FIX: Throughput_Gbps only — convert to Mbps for display
    metrics = [
        ("Avg_Latency_ms",   "Average Latency (ms)",  "latency"),
        ("Packet_Loss_Rate", "Packet Loss Rate",       "packet_loss"),
        ("Throughput_Gbps",  "Throughput (Mbps)",      "throughput"),
    ]

    for col, ylabel, fname in metrics:
        if col not in combined.columns:
            continue

        scale  = 1000.0 if col == "Throughput_Gbps" else 1.0
        means  = combined.groupby("Algorithm")[col].mean().reindex(algorithms) * scale
        stds   = combined.groupby("Algorithm")[col].std().reindex(algorithms).fillna(0) * scale

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(algorithms, means, yerr=stds, capsize=5,
                      color=colors, edgecolor="white",
                      error_kw={"elinewidth": 1.2})
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} — Mean ± Std over {len(seed_dfs)} seeds")
        ax.grid(True, axis="y", alpha=0.3)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2.,
                    bar.get_height() + s + max(means) * 0.01,
                    f"{m:.4f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"multiseed_{fname}.png"))
        plt.close()
        print(f"  Saved: multiseed_{fname}.png")


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------

def generate_all_plots(results_dir="results", output_dir="project_images"):
    print("=" * 60)
    print("GENERATING ALL PLOTS")
    print("=" * 60)

    print("\n[1] Training curves...")
    plot_training_curves(results_dir, output_dir)

    print("\n[2] Comparison charts...")
    plot_comparison_charts(results_dir, output_dir)

    print("\n[3] Network topology...")
    plot_network_topology(results_dir, output_dir)

    print("\n[4] Multi-seed summary (paper error bars)...")
    plot_multi_seed_summary(results_dir, output_dir)

    print(f"\nAll plots saved to: {output_dir}/")


if __name__ == "__main__":
    generate_all_plots()