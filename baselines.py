"""
Traditional Routing Baselines
===============================
Dijkstra's Algorithm and Distance Vector Routing.
These serve as baselines to show why DRL-based routing is needed.
Both run on the SAME network with the SAME traffic as MAPPO/MADDPG.

FIXES
-----
1. Metric key consistency: _get_episode_metrics() returns "avg_packet_loss"
   and "avg_throughput". After the evaluate() loop prefixes them with "avg_"
   they become "avg_avg_packet_loss" and "avg_avg_throughput" — matching
   exactly what evaluate_all.py's _get() helper expects.

2. Per-node action map replaces the old single scalar that was overwritten
   by every flow — only the last flow's action survived, misrouting all
   others → dropped → negative reward.
"""

import numpy as np
import networkx as nx
from collections import defaultdict
import time


# ---------------------------------------------------------------------------
# Helper shared by both routers
# ---------------------------------------------------------------------------

def _pick_action_for_agent(agent_id, env, node_to_nexthop_fn):
    """
    Build a per-node next-hop map for all nodes managed by agent_id, then
    return the action (neighbour index) for the first managed node that has
    at least one active, unfinished flow.
    Falls back to 0 only when the agent is genuinely idle.
    """
    managed_nodes   = env.agent_node_assignments[agent_id]
    node_action_map = {}

    for flow in env.active_flows:
        curr = flow["current_node"]
        if flow["delivered"] or flow["dropped"]:
            continue
        if curr not in managed_nodes:
            continue
        if curr in node_action_map:
            continue

        next_hop  = node_to_nexthop_fn(curr, flow["dst"])
        if next_hop is None:
            continue

        neighbors = list(env.graph.neighbors(curr))
        if next_hop in neighbors:
            node_action_map[curr] = neighbors.index(next_hop)

    for node in managed_nodes:
        if node in node_action_map:
            return int(node_action_map[node])

    return 0


# ---------------------------------------------------------------------------
# Dijkstra Router
# ---------------------------------------------------------------------------

class DijkstraRouter:
    """
    Dijkstra's Shortest Path Routing.
    Computes shortest path based on static propagation_delay weights.
    Cannot adapt to dynamic traffic — the intentional baseline limitation.
    """

    def __init__(self, env):
        self.env   = env
        self.graph = env.graph
        self.name  = "Dijkstra"
        self.shortest_paths = dict(
            nx.all_pairs_dijkstra_path(self.graph, weight="propagation_delay")
        )

    def route_flow(self, src, dst):
        try:
            return self.shortest_paths[src][dst]
        except KeyError:
            return None

    def get_next_hop(self, current_node, dst_node):
        path = self.route_flow(current_node, dst_node)
        if path and len(path) > 1:
            return path[1]
        return None

    def evaluate(self, num_episodes=200, load_multiplier=1.0):
        all_metrics       = defaultdict(list)
        computation_times = []

        for ep in range(num_episodes):
            obs, gs = self.env.reset()

            for _step in range(self.env.max_steps):
                t0      = time.time()
                actions = {
                    agent_id: _pick_action_for_agent(agent_id, self.env, self.get_next_hop)
                    for agent_id in range(self.env.num_agents)
                }
                computation_times.append(time.time() - t0)

                obs, gs, rewards, done, info = self.env.step(actions)
                if done:
                    break

            ep_m = self.env._get_episode_metrics()
            for key, val in ep_m.items():
                if isinstance(val, (int, float)):
                    all_metrics[key].append(val)

        results = {}
        for key, vals in all_metrics.items():
            results[f"avg_{key}"] = float(np.mean(vals))
            results[f"std_{key}"] = float(np.std(vals))

        results["avg_computation_time"] = float(np.mean(computation_times))
        results["algorithm"]            = self.name
        return results


# ---------------------------------------------------------------------------
# Distance Vector Router
# ---------------------------------------------------------------------------

class DistanceVectorRouter:
    """
    Distance Vector Routing (Bellman-Ford based).
    Hop-count metric, suffers from count-to-infinity — intentional baseline.
    """

    def __init__(self, env, max_iterations=100):
        self.env            = env
        self.graph          = env.graph
        self.name           = "DistanceVector"
        self.num_nodes      = env.num_nodes
        self.max_iterations = max_iterations
        self.routing_tables = self._build_routing_tables()

    def _build_routing_tables(self):
        tables = {}
        for node in self.graph.nodes():
            tables[node] = {node: (node, 0)}
            for nbr in self.graph.neighbors(node):
                tables[node][nbr] = (nbr, 1)

        for _ in range(self.max_iterations):
            updated = False
            for node in self.graph.nodes():
                for nbr in self.graph.neighbors(node):
                    for dest, (_nh, dist) in tables[nbr].items():
                        new_dist = dist + 1
                        if dest not in tables[node] or new_dist < tables[node][dest][1]:
                            tables[node][dest] = (nbr, new_dist)
                            updated = True
            if not updated:
                break

        return tables

    def get_next_hop(self, current_node, dst_node):
        entry = self.routing_tables.get(current_node, {}).get(dst_node)
        return entry[0] if entry is not None else None

    def evaluate(self, num_episodes=200, load_multiplier=1.0):
        all_metrics       = defaultdict(list)
        computation_times = []

        for ep in range(num_episodes):
            obs, gs = self.env.reset()

            for _step in range(self.env.max_steps):
                t0      = time.time()
                actions = {
                    agent_id: _pick_action_for_agent(agent_id, self.env, self.get_next_hop)
                    for agent_id in range(self.env.num_agents)
                }
                computation_times.append(time.time() - t0)

                obs, gs, rewards, done, info = self.env.step(actions)
                if done:
                    break

            ep_m = self.env._get_episode_metrics()
            for key, val in ep_m.items():
                if isinstance(val, (int, float)):
                    all_metrics[key].append(val)

        results = {}
        for key, vals in all_metrics.items():
            results[f"avg_{key}"] = float(np.mean(vals))
            results[f"std_{key}"] = float(np.std(vals))

        results["avg_computation_time"] = float(np.mean(computation_times))
        results["algorithm"]            = self.name
        return results


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from network_env import HealthcareSDNEnv

    env = HealthcareSDNEnv(num_nodes=30, num_agents=3, seed=42)

    print("Testing Dijkstra routing...")
    dijkstra  = DijkstraRouter(env)
    d_results = dijkstra.evaluate(num_episodes=10)
    print(f"  Avg Latency : {d_results.get('avg_avg_latency', 0):.4f} ms")
    print(f"  Packet Loss : {d_results.get('avg_avg_packet_loss', 0):.4f}")
    print(f"  Throughput  : {d_results.get('avg_avg_throughput', 0) * 1000:.4f} Mbps")

    print("\nTesting Distance Vector routing...")
    dv         = DistanceVectorRouter(env)
    dv_results = dv.evaluate(num_episodes=10)
    print(f"  Avg Latency : {dv_results.get('avg_avg_latency', 0):.4f} ms")
    print(f"  Packet Loss : {dv_results.get('avg_avg_packet_loss', 0):.4f}")
    print(f"  Throughput  : {dv_results.get('avg_avg_throughput', 0) * 1000:.4f} Mbps")

    print("\nBaseline tests passed!")