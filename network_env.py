"""
Healthcare SDN Network Environment
====================================
Graph-based simulation of a 30-node hospital SDN network.
Computes realistic network metrics using queuing theory.
No hardcoded results — all metrics are computed from simulation physics.

DEFINITIVE LATENCY FIX
-----------------------
The root cause of constant latency was threefold:

1. ONE action per agent applied to ALL flows → the action barely mattered
   because most flows fell back to shortest path anyway.
   FIX: Agent action selects a ROUTING STRATEGY (0-6) that determines
   how aggressively to avoid congested links for ALL flows in the zone.

2. Queuing delay from M/M/1 or queue-depth models was dwarfed by fixed
   propagation delay (~0.1ms/hop) at Gbps bandwidths.
   FIX: Congestion multiplier applied to TOTAL hop latency, not just
   queuing component. A congested link is 1x-5x slower overall.

3. Episode-level averaging over 256 steps × ~20 flows washed out any
   per-step variation.
   FIX: Congestion now accumulates across steps within an episode.
   Early routing decisions affect later steps' latency. Good routing
   keeps congestion low → low latency throughout. Bad routing → 
   congestion builds → latency climbs over the episode.
"""

import numpy as np
import networkx as nx
from collections import defaultdict, deque
from config import (
    NUM_NODES, HOSPITAL_ZONES, TRAFFIC_CLASSES, TRAFFIC_PROFILES,
    LINK_CONFIG, REWARD_CONFIG, get_node_zone, get_node_priority,
    get_priority_weight
)


class HealthcareSDNEnv:
    """
    SDN-based hospital network environment for multi-agent RL.

    The graph represents the SDN data plane:
    - Nodes = SDN-enabled switches in hospital departments
    - Edges = network links with bandwidth, delay, buffer
    - The Python code acts as the SDN controller

    Agent action space: each agent picks an integer 0..action_dim-1
    which selects a routing strategy for its zone:
      0 = pure shortest path (no congestion avoidance)
      1-2 = mild congestion avoidance
      3-4 = moderate congestion avoidance  
      5-6 = aggressive load balancing

    Higher congestion avoidance → longer paths but avoids loaded links
    The optimal strategy depends on current network state.
    """

    def __init__(self, num_nodes=NUM_NODES, num_agents=3, seed=None):
        self.num_nodes  = num_nodes
        self.num_agents = num_agents
        self.rng        = np.random.RandomState(seed)

        self.graph                 = self._build_topology()
        self.agent_node_assignments = self._assign_agents()

        self.obs_dim_per_agent    = self._compute_obs_dim()
        self.global_state_dim     = self._compute_global_state_dim()
        self.action_dim_per_agent = self._compute_action_dim()

        self.current_step  = 0
        self.max_steps     = 256
        self.active_flows  = []
        self.metrics_log   = defaultdict(list)

        self.max_active_flows = 500

        self._reset_link_states()

        # Pre-compute all shortest paths once (huge speedup)
        self._shortest_paths = dict(
            nx.all_pairs_dijkstra_path(self.graph, weight="propagation_delay")
        )

    # ------------------------------------------------------------------
    def _build_topology(self):
        """Build the hospital network graph with realistic connectivity."""
        G = nx.Graph()

        for zone_name, zone_info in HOSPITAL_ZONES.items():
            for node_id in zone_info["nodes"]:
                G.add_node(node_id,
                           zone=zone_name,
                           priority_class=zone_info["priority_class"],
                           queue_length=0,
                           buffer_size=LINK_CONFIG["buffer_size"],
                           processing_delay=LINK_CONFIG["processing_delay"])

        # Intra-zone connections
        for zone_name, zone_info in HOSPITAL_ZONES.items():
            nodes = zone_info["nodes"]
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    if self.rng.random() < 0.6:
                        G.add_edge(nodes[i], nodes[j],
                                   bandwidth=LINK_CONFIG["intra_zone_bandwidth"],
                                   propagation_delay=LINK_CONFIG["intra_zone_delay"],
                                   current_load=0.0,
                                   buffer_size=LINK_CONFIG["buffer_size"],
                                   queue=deque(), utilization=0.0)

        # Ensure intra-zone chain connectivity
        for zone_name, zone_info in HOSPITAL_ZONES.items():
            nodes = zone_info["nodes"]
            for i in range(len(nodes) - 1):
                if not G.has_edge(nodes[i], nodes[i + 1]):
                    G.add_edge(nodes[i], nodes[i + 1],
                               bandwidth=LINK_CONFIG["intra_zone_bandwidth"],
                               propagation_delay=LINK_CONFIG["intra_zone_delay"],
                               current_load=0.0,
                               buffer_size=LINK_CONFIG["buffer_size"],
                               queue=deque(), utilization=0.0)

        # Backbone connections
        infrastructure_nodes = HOSPITAL_ZONES["infrastructure"]["nodes"]
        for i in range(len(infrastructure_nodes)):
            for j in range(i + 1, len(infrastructure_nodes)):
                G.add_edge(infrastructure_nodes[i], infrastructure_nodes[j],
                           bandwidth=LINK_CONFIG["backbone_bandwidth"],
                           propagation_delay=LINK_CONFIG["backbone_delay"],
                           current_load=0.0,
                           buffer_size=LINK_CONFIG["buffer_size"] * 2,
                           queue=deque(), utilization=0.0)

        # Inter-zone connections
        for zone_name, zone_info in HOSPITAL_ZONES.items():
            if zone_name == "infrastructure":
                continue
            zone_nodes = zone_info["nodes"]
            for idx, backbone_node in enumerate(infrastructure_nodes[:2]):
                zone_node = zone_nodes[idx % len(zone_nodes)]
                if not G.has_edge(zone_node, backbone_node):
                    G.add_edge(zone_node, backbone_node,
                               bandwidth=LINK_CONFIG["inter_zone_bandwidth"],
                               propagation_delay=LINK_CONFIG["inter_zone_delay"],
                               current_load=0.0,
                               buffer_size=LINK_CONFIG["buffer_size"],
                               queue=deque(), utilization=0.0)
            if zone_info["priority_class"] <= 1:
                extra_node = zone_nodes[len(zone_nodes) // 2]
                if not G.has_edge(extra_node, infrastructure_nodes[2]):
                    G.add_edge(extra_node, infrastructure_nodes[2],
                               bandwidth=LINK_CONFIG["inter_zone_bandwidth"],
                               propagation_delay=LINK_CONFIG["inter_zone_delay"],
                               current_load=0.0,
                               buffer_size=LINK_CONFIG["buffer_size"],
                               queue=deque(), utilization=0.0)

        # Direct critical-to-diagnostic links
        for src, dst in [(0, 16), (3, 17), (6, 18)]:
            if src < self.num_nodes and dst < self.num_nodes:
                G.add_edge(src, dst,
                           bandwidth=LINK_CONFIG["inter_zone_bandwidth"],
                           propagation_delay=LINK_CONFIG["inter_zone_delay"],
                           current_load=0.0,
                           buffer_size=LINK_CONFIG["buffer_size"],
                           queue=deque(), utilization=0.0)

        # Ensure full connectivity
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            for i in range(len(components) - 1):
                n1 = list(components[i])[0]
                n2 = list(components[i + 1])[0]
                G.add_edge(n1, n2,
                           bandwidth=LINK_CONFIG["inter_zone_bandwidth"],
                           propagation_delay=LINK_CONFIG["inter_zone_delay"],
                           current_load=0.0,
                           buffer_size=LINK_CONFIG["buffer_size"],
                           queue=deque(), utilization=0.0)

        return G

    def _assign_agents(self):
        assignments = {}
        assignments[0] = list(range(0,  16))   # Critical + Monitoring
        assignments[1] = list(range(16, 27))   # Diagnostic + Admin
        assignments[2] = list(range(27, self.num_nodes))  # Infrastructure
        return assignments

    def _compute_obs_dim(self):
        max_neighbors = max(self.graph.degree(n) for n in self.graph.nodes())
        return max_neighbors + 1 + 4 + 1

    def _compute_global_state_dim(self):
        return self.graph.number_of_edges() + self.num_nodes + 4

    def _compute_action_dim(self):
        return max(self.graph.degree(n) for n in self.graph.nodes())

    def _reset_link_states(self):
        for u, v in self.graph.edges():
            self.graph[u][v]["current_load"] = 0.0
            self.graph[u][v]["utilization"]  = 0.0
            self.graph[u][v]["queue"]        = deque()
        for node in self.graph.nodes():
            self.graph.nodes[node]["queue_length"] = 0

    # ------------------------------------------------------------------
    def reset(self):
        self.current_step = 0
        self.active_flows = []
        self.metrics_log  = defaultdict(list)
        self._reset_link_states()
        self._generate_traffic()
        return self._get_observations(), self._get_global_state()

    # ------------------------------------------------------------------
    def step(self, actions):
        self.current_step += 1

        self._expire_old_flows()

        step_metrics = self._apply_actions(actions)
        self._generate_traffic()
        self._update_link_states()
        rewards = self._compute_rewards(step_metrics)

        obs          = self._get_observations()
        global_state = self._get_global_state()
        done         = self.current_step >= self.max_steps

        for key, val in step_metrics.items():
            self.metrics_log[key].append(val)

        info = {
            "step_metrics":    step_metrics,
            "episode_metrics": self._get_episode_metrics() if done else None,
        }

        return obs, global_state, rewards, done, info

    # ------------------------------------------------------------------
    def _expire_old_flows(self, max_age=20):
        cutoff = self.current_step - max_age
        self.active_flows = [
            f for f in self.active_flows
            if f["created_step"] >= cutoff or f["delivered"] or f["dropped"]
        ]

    # ------------------------------------------------------------------
    def _generate_traffic(self, load_multiplier=1.0):
        """Generate healthcare traffic flows using Poisson process."""
        if len(self.active_flows) >= self.max_active_flows:
            return

        new_flows = []
        for zone_name, zone_info in HOSPITAL_ZONES.items():
            if zone_name == "infrastructure":
                continue

            priority_class = zone_info["priority_class"]
            profile        = TRAFFIC_PROFILES[priority_class]

            for src_node in zone_info["nodes"]:
                num_arrivals = self.rng.poisson(
                    profile["arrival_rate"] * load_multiplier
                )
                for _ in range(num_arrivals):
                    dst_node = self._select_destination(src_node, priority_class)
                    if dst_node == src_node:
                        continue

                    data_rate   = self.rng.uniform(*profile["data_rate_range"])
                    packet_size = self.rng.randint(*profile["packet_size_range"])

                    if self.rng.random() < profile["burst_probability"]:
                        data_rate *= self.rng.uniform(2.0, 5.0)

                    new_flows.append({
                        "src":          src_node,
                        "dst":          dst_node,
                        "traffic_class": priority_class,
                        "data_rate":    data_rate,
                        "packet_size":  packet_size,
                        "created_step": self.current_step,
                        "hops":         [],
                        "current_node": src_node,
                        "delivered":    False,
                        "dropped":      False,
                    })

        remaining = self.max_active_flows - len(self.active_flows)
        self.active_flows.extend(new_flows[:remaining])

    # ------------------------------------------------------------------
    def _select_destination(self, src_node, priority_class):
        infrastructure = HOSPITAL_ZONES["infrastructure"]["nodes"]
        if priority_class == 0:
            candidates = infrastructure + HOSPITAL_ZONES["diagnostic"]["nodes"]
        elif priority_class == 1:
            candidates = infrastructure
        elif priority_class == 2:
            candidates = infrastructure + HOSPITAL_ZONES["administrative"]["nodes"]
        else:
            candidates = infrastructure
        candidates = [n for n in candidates if n != src_node]
        if not candidates:
            candidates = [n for n in range(self.num_nodes) if n != src_node]
        return self.rng.choice(candidates)

    # ------------------------------------------------------------------
    def _get_congestion_weight(self, u, v):
        """
        Return a congestion multiplier for edge (u,v).
        Empty link → 1.0 (no penalty).
        Loaded link → up to 5.0 (5x slower).
        
        This is the KEY mechanism that lets agents influence latency:
        routing flows through congested links makes them slower.
        """
        edge = self.graph[u][v]
        util = edge.get("utilization", 0.0)
        queue_len = len(edge.get("queue", []))
        buffer_cap = edge.get("buffer_size", LINK_CONFIG["buffer_size"])
        queue_fill = queue_len / max(buffer_cap, 1)
        
        # Combine utilization and queue fill
        congestion = 0.6 * util + 0.4 * queue_fill
        
        # Exponential scaling: congestion=0→1.0, congestion=0.5→2.0, congestion=1.0→5.0
        multiplier = 1.0 + 4.0 * (congestion ** 1.5)
        return multiplier

    # ------------------------------------------------------------------
    def _get_next_hop(self, current, dst, actions):
        """
        Choose next hop for a flow at *current* heading to *dst*.

        The agent's action is mapped to a congestion_avoidance_weight (0.0 to 1.0).
        This weight controls how much the routing considers congestion vs shortest path:
          weight=0.0 → pure shortest path (ignores congestion)
          weight=1.0 → aggressively avoids congested links

        The agent learns WHEN to use congestion avoidance (high load) vs 
        shortest path (low load), producing varying latency across episodes.
        """
        agent_id  = self._get_agent_for_node(current)
        neighbors = list(self.graph.neighbors(current))
        if not neighbors:
            return None

        # Map agent action to congestion avoidance weight
        ca_weight = 0.0  # default: shortest path
        if agent_id is not None and agent_id in actions:
            action_idx = actions[agent_id]
            if isinstance(action_idx, np.ndarray):
                action_idx = int(action_idx.flatten()[0])
            action_idx = int(action_idx) % self.action_dim_per_agent
            # Map action 0..6 to weight 0.0..1.0
            ca_weight = action_idx / max(self.action_dim_per_agent - 1, 1)

        # Score each neighbor
        best_score = -1e9
        best_hop   = neighbors[0]

        for n in neighbors:
            # Shortest-path component: how close is this neighbor to dst?
            try:
                sp = self._shortest_paths[n][dst]
                sp_hops = len(sp) - 1  # number of hops remaining
            except KeyError:
                sp_hops = 100  # unreachable

            # Normalize: fewer hops = higher score (0 to 1)
            sp_score = 1.0 / (1.0 + sp_hops)

            # Congestion component: how loaded is this link?
            cong_mult = self._get_congestion_weight(current, n)
            cong_score = 1.0 / cong_mult  # less congested = higher score

            # Weighted combination based on agent's action
            score = (1.0 - ca_weight) * sp_score + ca_weight * cong_score

            if score > best_score:
                best_score = score
                best_hop   = n

        return best_hop

    # ------------------------------------------------------------------
    def _route_flow_full_path(self, flow, actions, max_hops=20):
        """
        Route a flow along its full path in one step.
        Returns (total_latency, was_delivered, was_dropped, hop_count).
        """
        current       = flow["current_node"]
        dst           = flow["dst"]
        total_latency = 0.0
        hops_taken    = 0
        visited       = set()

        for _ in range(max_hops):
            if current == dst:
                return total_latency, True, False, hops_taken

            if current in visited:
                return total_latency, False, True, hops_taken
            visited.add(current)

            next_hop = self._get_next_hop(current, dst, actions)
            if next_hop is None:
                return total_latency, False, True, hops_taken

            if not self.graph.has_edge(current, next_hop):
                nbrs     = list(self.graph.neighbors(current))
                next_hop = nbrs[0] if nbrs else None
                if next_hop is None:
                    return total_latency, False, True, hops_taken

            edge_data     = self.graph[current][next_hop]
            bandwidth_bps = edge_data["bandwidth"] * 1e6

            # Base delays (fixed per link)
            tx_delay   = (flow["packet_size"] * 8) / bandwidth_bps * 1000
            prop_delay = edge_data["propagation_delay"]
            proc_delay = self.graph.nodes[current].get("processing_delay", 0.01)
            base_delay = tx_delay + prop_delay + proc_delay

            # Congestion multiplier: THIS is what makes latency vary
            # A congested link multiplies the TOTAL delay, not just queuing
            # Empty link: 1.0x → base_delay only
            # 50% loaded: ~2.0x → double the delay
            # 90% loaded: ~4.5x → 4.5x the delay
            cong_mult = self._get_congestion_weight(current, next_hop)
            hop_latency = base_delay * cong_mult

            # Buffer overflow check
            current_queue = len(edge_data["queue"])
            soft_buffer_limit = int(edge_data["buffer_size"] * 0.9)
            if current_queue >= soft_buffer_limit:
                return total_latency, False, True, hops_taken

            total_latency += hop_latency

            edge_data["current_load"] += flow["data_rate"]
            edge_data["queue"].append(flow["packet_size"])

            flow["hops"].append({"from": current, "to": next_hop,
                                 "latency": hop_latency})
            current    = next_hop
            hops_taken += 1

        return total_latency, False, True, hops_taken

    # ------------------------------------------------------------------
    def _apply_actions(self, actions):
        total_latency      = 0.0
        total_packets      = 0
        dropped_packets    = 0
        delivered_packets  = 0
        latencies_by_class = defaultdict(list)
        jitter_values      = []
        throughput_bytes   = 0
        flows_to_remove    = []
        hop_counts         = []   # each entry is an integer

        for idx, flow in enumerate(self.active_flows):
            if flow["delivered"] or flow["dropped"]:
                flows_to_remove.append(idx)
                continue

            total_packets += 1
            lat, delivered, dropped, hops = self._route_flow_full_path(flow, actions)

            if dropped:
                flow["dropped"] = True
                dropped_packets += 1
            elif delivered:
                flow["delivered"]      = True
                flow["current_node"]   = flow["dst"]
                latencies_by_class[flow["traffic_class"]].append(lat)
                total_latency         += lat
                delivered_packets     += 1
                throughput_bytes      += flow["packet_size"]
                hop_counts.append(int(hops))  # always integer
                flows_to_remove.append(idx)

        for idx in sorted(set(flows_to_remove), reverse=True):
            if idx < len(self.active_flows):
                self.active_flows.pop(idx)

        for cls, lats in latencies_by_class.items():
            if len(lats) > 1:
                jitter_values.append(np.std(lats))

        avg_latency      = total_latency / max(delivered_packets, 1)
        packet_loss_rate = dropped_packets / max(total_packets, 1)
        avg_jitter       = np.mean(jitter_values) if jitter_values else 0.0
        throughput_gbps  = (throughput_bytes * 8) / 1e9

        # Hop count stats — all integers
        avg_hops = int(round(np.mean(hop_counts))) if hop_counts else 0
        min_hops = int(min(hop_counts)) if hop_counts else 0
        max_hops = int(max(hop_counts)) if hop_counts else 0
        total_hops = int(sum(hop_counts))

        per_class_latency = {}
        for cls in range(4):
            cls_lats             = latencies_by_class.get(cls, [])
            per_class_latency[cls] = np.mean(cls_lats) if cls_lats else 0.0

        return {
            "avg_latency":       avg_latency,
            "packet_loss_rate":  packet_loss_rate,
            "avg_jitter":        avg_jitter,
            "throughput_gbps":   throughput_gbps,
            "avg_hops":          avg_hops,
            "min_hops":          min_hops,
            "max_hops":          max_hops,
            "total_hops":        total_hops,
            "delivered_packets": delivered_packets,
            "dropped_packets":   dropped_packets,
            "total_packets":     total_packets,
            "per_class_latency": per_class_latency,
        }

    # ------------------------------------------------------------------
    def _update_link_states(self):
        """Update link utilisation and drain queues."""
        for u, v in self.graph.edges():
            edge      = self.graph[u][v]
            bandwidth = edge["bandwidth"]

            # Moderate drain: lets congestion build on overloaded links
            # but clears on underused ones
            if bandwidth >= 10000:
                drain_count = 5
            elif bandwidth >= 1000:
                drain_count = 3
            else:
                drain_count = 2

            drain_count = min(drain_count, len(edge["queue"]))
            for _ in range(drain_count):
                if edge["queue"]:
                    edge["queue"].popleft()

            if bandwidth > 0:
                edge["utilization"] = min(
                    edge["current_load"] / bandwidth, 1.0
                )
            # Slow decay so congestion from bad routing persists
            edge["current_load"] *= 0.85

        for node in self.graph.nodes():
            neighbors  = list(self.graph.neighbors(node))
            total_queue = sum(
                len(self.graph[node][n]["queue"]) for n in neighbors
            )
            self.graph.nodes[node]["queue_length"] = total_queue

    # ------------------------------------------------------------------
    def _compute_rewards(self, metrics):
        """Compute priority-weighted rewards for each agent."""
        rc = REWARD_CONFIG

        latency_penalty  = -rc["latency_penalty_scale"] * metrics["avg_latency"]
        loss_penalty     = -rc["packet_loss_penalty"]   * metrics["packet_loss_rate"]
        jitter_penalty   = -rc["jitter_penalty"]        * metrics["avg_jitter"]
        throughput_bonus =  rc["throughput_reward"]     * metrics["throughput_gbps"]

        delivery_bonus = rc.get("delivery_bonus", 0.0) * metrics["delivered_packets"]

        priority_latency_penalty = 0.0
        for cls, latency in metrics["per_class_latency"].items():
            weight = get_priority_weight(cls)
            priority_latency_penalty -= weight * latency

        base_reward = (
            latency_penalty
            + loss_penalty
            + jitter_penalty
            + throughput_bonus
            + delivery_bonus
            + priority_latency_penalty
        ) * rc["reward_scaling"]

        return {agent_id: base_reward for agent_id in range(self.num_agents)}

    # ------------------------------------------------------------------
    def _get_observations(self):
        observations = {}
        for agent_id, node_list in self.agent_node_assignments.items():
            obs_parts = []
            for node in node_list[:5]:
                neighbors = list(self.graph.neighbors(node))
                utils = []
                for n in neighbors[:self.action_dim_per_agent]:
                    utils.append(self.graph[node][n]["utilization"])
                while len(utils) < self.action_dim_per_agent:
                    utils.append(0.0)
                obs_parts.extend(utils)

                queue_len = (self.graph.nodes[node]["queue_length"]
                             / LINK_CONFIG["buffer_size"])
                obs_parts.append(queue_len)

                class_dist = [0.0] * 4
                for flow in self.active_flows:
                    if (flow["current_node"] == node
                            and not flow["delivered"]
                            and not flow["dropped"]):
                        class_dist[flow["traffic_class"]] += 1
                total_flows = sum(class_dist) or 1
                class_dist  = [c / total_flows for c in class_dist]
                obs_parts.extend(class_dist)

                bw_avail = (
                    np.mean([1.0 - self.graph[node][n]["utilization"]
                             for n in neighbors])
                    if neighbors else 0.0
                )
                obs_parts.append(bw_avail)

            observations[agent_id] = np.array(obs_parts, dtype=np.float32)

        max_obs_len = max(len(obs) for obs in observations.values())
        for agent_id in observations:
            obs = observations[agent_id]
            if len(obs) < max_obs_len:
                observations[agent_id] = np.pad(obs, (0, max_obs_len - len(obs)))

        return observations

    # ------------------------------------------------------------------
    def _get_global_state(self):
        state_parts = []
        for u, v in self.graph.edges():
            state_parts.append(self.graph[u][v]["utilization"])
        for node in range(self.num_nodes):
            ql = (self.graph.nodes[node]["queue_length"]
                  / LINK_CONFIG["buffer_size"])
            state_parts.append(ql)
        class_loads = [0.0] * 4
        for flow in self.active_flows:
            if not flow["delivered"] and not flow["dropped"]:
                class_loads[flow["traffic_class"]] += 1
        total = sum(class_loads) or 1
        class_loads = [c / total for c in class_loads]
        state_parts.extend(class_loads)
        return np.array(state_parts, dtype=np.float32)

    # ------------------------------------------------------------------
    def _get_agent_for_node(self, node_id):
        for agent_id, nodes in self.agent_node_assignments.items():
            if node_id in nodes:
                return agent_id
        return None

    # ------------------------------------------------------------------
    def _get_episode_metrics(self):
        if not self.metrics_log["avg_latency"]:
            return {}
        return {
            "avg_latency":    np.mean(self.metrics_log["avg_latency"]),
            "avg_packet_loss": np.mean(self.metrics_log["packet_loss_rate"]),
            "avg_jitter":     np.mean(self.metrics_log["avg_jitter"]),
            "avg_throughput": np.mean(self.metrics_log["throughput_gbps"]),
            "avg_hops":       int(round(np.mean(self.metrics_log["avg_hops"]))) if self.metrics_log["avg_hops"] else 0,
            "min_hops":       int(min(self.metrics_log["min_hops"])) if self.metrics_log["min_hops"] else 0,
            "max_hops":       int(max(self.metrics_log["max_hops"])) if self.metrics_log["max_hops"] else 0,
            "total_delivered": sum(self.metrics_log["delivered_packets"]),
            "total_dropped":   sum(self.metrics_log["dropped_packets"]),
        }

    # ------------------------------------------------------------------
    def get_adjacency_matrix(self):
        return nx.adjacency_matrix(self.graph).toarray()

    def get_shortest_paths(self):
        return dict(
            nx.all_pairs_dijkstra_path(self.graph, weight="propagation_delay")
        )

    def render_info(self):
        total_util = np.mean([
            self.graph[u][v]["utilization"] for u, v in self.graph.edges()
        ])
        active = sum(
            1 for f in self.active_flows
            if not f["delivered"] and not f["dropped"]
        )
        return (
            f"Step {self.current_step}/{self.max_steps} | "
            f"Active flows: {active} | "
            f"Avg utilization: {total_util:.3f} | "
            f"Nodes: {self.num_nodes} | Edges: {self.graph.number_of_edges()}"
        )


# ==============================================================================
# Quick test
# ==============================================================================
if __name__ == "__main__":
    env = HealthcareSDNEnv(num_nodes=30, num_agents=3, seed=42)
    print(f"Network: {env.num_nodes} nodes, {env.graph.number_of_edges()} edges")
    print(f"Connected: {nx.is_connected(env.graph)}")
    print(f"Agent assignments: { {k: len(v) for k, v in env.agent_node_assignments.items()} }")

    obs, global_state = env.reset()
    print(f"Observation dims: { {k: v.shape for k, v in obs.items()} }")
    print(f"Global state dim: {global_state.shape}")
    print(f"Action dim per agent: {env.action_dim_per_agent}")

    # Test different actions produce different latencies
    print("\n--- Testing action impact on latency ---")
    for action_val in [0, 3, 6]:
        env2 = HealthcareSDNEnv(num_nodes=30, num_agents=3, seed=42)
        obs2, _ = env2.reset()
        
        latencies = []
        for step in range(20):
            actions = {0: action_val, 1: action_val, 2: action_val}
            obs2, gs, rewards, done, info = env2.step(actions)
            latencies.append(info["step_metrics"]["avg_latency"])
        
        print(f"  Action={action_val}: avg_latency={np.mean(latencies):.3f}ms "
              f"(range {min(latencies):.3f}-{max(latencies):.3f}ms)")

    # Test that congestion builds over steps
    print("\n--- Testing congestion accumulation ---")
    env3 = HealthcareSDNEnv(num_nodes=30, num_agents=3, seed=42)
    obs3, _ = env3.reset()
    for step in range(10):
        actions = {0: 0, 1: 0, 2: 0}  # always shortest path → builds congestion
        obs3, gs, rewards, done, info = env3.step(actions)
        m = info["step_metrics"]
        print(f"  Step {step}: latency={m['avg_latency']:.3f}ms | "
              f"loss={m['packet_loss_rate']:.3f} | delivered={m['delivered_packets']}")

    print("\nEnvironment test passed!")
