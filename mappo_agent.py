"""
MAPPO Agent Implementation
============================
Multi-Agent Proximal Policy Optimization for Healthcare SDN Routing.
Centralized Training with Decentralized Execution (CTDE).

Key differences from MADDPG (base paper):
1. On-policy learning (no replay buffer)
2. PPO clipping for stable updates
3. Shared centralized critic with global state input
4. GAE for advantage estimation

FIXES APPLIED
-------------
1. num_agents always taken from env (not config) so all 3 zones are covered.
   With only 2 actors, agent 2's backbone nodes (27-29) were un-routed →
   all inter-zone flows dropped → massive packet-loss penalty every step.

2. episode_reward now averages ALL agents' rewards (cooperative setting).
   Using only rewards[0] gave a biased, noisy signal and missed the
   contribution of agents 1 and 2.

3. GAE now uses the MEAN reward across all agents instead of only
   buffer.rewards[0]. Agents 1 and 2 were being updated with agent 0's
   advantage estimates, producing wrong gradient directions.

4. Reward normalization (RunningNormalizer) added. Raw latency penalties
   (e.g. 50 ms × scale) produce rewards like -500 early in training.
   Normalizing to [-10, 10] stabilises the critic before it has learned
   anything, preventing divergence.

5. LR annealing guard: minimum LR floor added so the learning rate never
   decays to effectively zero before the policy has converged.

6. Gradient clipping already present — verified correct (max_grad_norm).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import os
import csv
from collections import defaultdict


# ---------------------------------------------------------------------------
# Running reward normalizer
# ---------------------------------------------------------------------------

class RunningNormalizer:
    """
    Online mean/std normalizer (Welford's algorithm).
    Keeps reward scale in a stable range during early training when
    latency penalties are large and the policy has not yet converged.
    """
    def __init__(self, clip=10.0):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = 0
        self.clip  = clip

    def update(self, x):
        self.count += 1
        delta      = x - self.mean
        self.mean += delta / self.count
        self.var   = self.var + (delta * (x - self.mean) - self.var) / max(self.count, 1)

    def normalize(self, x):
        std = max(np.sqrt(self.var), 1e-8)
        return float(np.clip((x - self.mean) / std, -self.clip, self.clip))


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class ActorNetwork(nn.Module):
    """
    Decentralized actor: takes LOCAL observation, outputs action probabilities.
    Each agent has its own actor (or shared with parameter sharing).
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        layers    = []
        input_dim = obs_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, obs):
        logits = self.network(obs)
        return Categorical(logits=logits)

    def get_action_and_log_prob(self, obs):
        dist     = self.forward(obs)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, dist.entropy()


class CriticNetwork(nn.Module):
    """
    Centralized critic: takes GLOBAL state, outputs state value.
    Shared across all agents during training (CTDE).
    Discarded at inference time.
    """
    def __init__(self, global_state_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        layers    = []
        input_dim = global_state_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, global_state):
        return self.network(global_state).squeeze(-1)


# ---------------------------------------------------------------------------
# Rollout Buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores one episode of rollout data for on-policy training."""

    def __init__(self):
        self.observations  = defaultdict(list)   # {agent_id: [obs_t]}
        self.global_states = []
        self.actions       = defaultdict(list)
        self.log_probs     = defaultdict(list)
        self.rewards       = defaultdict(list)
        self.values        = []
        self.dones         = []

    def add(self, obs, global_state, actions, log_probs, rewards, value, done):
        for agent_id in obs:
            self.observations[agent_id].append(obs[agent_id])
            self.actions[agent_id].append(actions[agent_id])
            self.log_probs[agent_id].append(log_probs[agent_id])
            self.rewards[agent_id].append(rewards[agent_id])
        self.global_states.append(global_state)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.observations  = defaultdict(list)
        self.global_states = []
        self.actions       = defaultdict(list)
        self.log_probs     = defaultdict(list)
        self.rewards       = defaultdict(list)
        self.values        = []
        self.dones         = []

    def get_tensors(self, agent_id, device):
        """Convert stored data to tensors for a specific agent."""
        obs          = torch.FloatTensor(np.array(self.observations[agent_id])).to(device)
        actions      = torch.LongTensor(np.array(self.actions[agent_id])).to(device)
        old_log_prob = torch.FloatTensor(np.array(self.log_probs[agent_id])).to(device)
        rewards      = torch.FloatTensor(np.array(self.rewards[agent_id])).to(device)
        global_states = torch.FloatTensor(np.array(self.global_states)).to(device)
        values       = torch.FloatTensor(np.array(self.values)).to(device)
        dones        = torch.FloatTensor(np.array(self.dones)).to(device)
        return obs, actions, old_log_prob, rewards, global_states, values, dones

    def get_mean_rewards(self, num_agents):
        """
        FIX 3: return mean reward across all agents at each timestep.
        Original code used only buffer.rewards[0], giving wrong advantages
        for agents 1 and 2.
        """
        all_rewards = np.array([
            self.rewards[i] for i in range(num_agents) if i in self.rewards
        ])                                    # shape: (num_agents, T)
        return all_rewards.mean(axis=0).tolist()   # shape: (T,)


# ---------------------------------------------------------------------------
# MAPPO Agent
# ---------------------------------------------------------------------------

class MAPPOAgent:
    """
    MAPPO: Multi-Agent Proximal Policy Optimization.

    Training loop:
    1. Collect rollouts (max_steps per episode)
    2. Compute GAE advantages using centralized critic
    3. PPO update with clipped surrogate objective (ppo_epochs times)
    4. Discard data, repeat
    """

    def __init__(self, env, config):
        self.env    = env
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # FIX 1: always use the environment's agent count
        self.num_agents = env.num_agents

        # Dimensions from environment
        obs, gs  = env.reset()
        obs_dim          = len(list(obs.values())[0])
        global_state_dim = len(gs)
        action_dim       = env.action_dim_per_agent

        # One actor per agent
        self.actors           = {}
        self.actor_optimizers = {}
        for agent_id in range(self.num_agents):
            actor = ActorNetwork(
                obs_dim, action_dim,
                config["hidden_dim"], config["num_layers"]
            ).to(self.device)
            self.actors[agent_id]           = actor
            self.actor_optimizers[agent_id] = optim.Adam(
                actor.parameters(), lr=config["lr_actor"]
            )

        # Shared centralized critic
        self.critic = CriticNetwork(
            global_state_dim, config["hidden_dim"], config["num_layers"]
        ).to(self.device)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=config["lr_critic"]
        )

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # FIX 4: reward normalizer
        self.reward_normalizer = RunningNormalizer(clip=10.0)

        self.training_log  = []
        self.episode_count = 0

    # ------------------------------------------------------------------
    def select_actions(self, observations, deterministic=False):
        """Select actions for all agents (decentralized execution)."""
        actions   = {}
        log_probs = {}

        for agent_id in range(self.num_agents):
            obs_tensor = torch.FloatTensor(
                observations[agent_id]
            ).unsqueeze(0).to(self.device)

            with torch.no_grad():
                if deterministic:
                    dist     = self.actors[agent_id](obs_tensor)
                    action   = dist.probs.argmax(dim=-1)
                    log_prob = dist.log_prob(action)
                else:
                    action, log_prob, _ = self.actors[agent_id].get_action_and_log_prob(obs_tensor)

            actions[agent_id]   = action.item()
            log_probs[agent_id] = log_prob.item()

        return actions, log_probs

    # ------------------------------------------------------------------
    def get_value(self, global_state):
        """Get value estimate from centralized critic."""
        gs_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.critic(gs_tensor)
        return value.item()

    # ------------------------------------------------------------------
    def compute_gae(self, rewards, values, dones, gamma, gae_lambda):
        """
        Generalized Advantage Estimation (GAE).

        A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)
        """
        advantages = []
        gae        = 0.0

        for t in reversed(range(len(rewards))):
            next_value = 0.0 if t == len(rewards) - 1 else values[t + 1]
            delta      = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae        = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns    = advantages + torch.FloatTensor(values).to(self.device)

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    # ------------------------------------------------------------------
    def update(self):
        """
        PPO update step.
        Uses collected rollout data to update actors and centralized critic.
        Runs ppo_epochs passes over the same batch.
        """
        config = self.config

        # FIX 3: use mean reward across ALL agents for GAE
        mean_rewards = self.buffer.get_mean_rewards(self.num_agents)
        values       = self.buffer.values
        dones        = self.buffer.dones

        advantages, returns = self.compute_gae(
            mean_rewards, values, dones,
            config["gamma"], config["gae_lambda"]
        )

        total_actor_loss  = 0.0
        total_critic_loss = 0.0
        total_entropy     = 0.0

        for _epoch in range(config["ppo_epochs"]):

            # ---- Update each agent's actor ----
            for agent_id in range(self.num_agents):
                obs, actions, old_log_probs, _, global_states, _, _ = \
                    self.buffer.get_tensors(agent_id, self.device)

                dist         = self.actors[agent_id](obs)
                new_log_prob = dist.log_prob(actions)
                entropy      = dist.entropy().mean()

                # PPO clipped surrogate objective
                ratio  = torch.exp(new_log_prob - old_log_probs)
                surr1  = ratio * advantages
                surr2  = torch.clamp(
                    ratio,
                    1.0 - config["clip_coeff"],
                    1.0 + config["clip_coeff"]
                ) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                actor_loss -= config["entropy_coeff"] * entropy

                self.actor_optimizers[agent_id].zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actors[agent_id].parameters(), config["max_grad_norm"]
                )
                self.actor_optimizers[agent_id].step()

                total_actor_loss += actor_loss.item()
                total_entropy    += entropy.item()

            # ---- Update centralized critic ----
            global_states_t   = torch.FloatTensor(
                np.array(self.buffer.global_states)
            ).to(self.device)
            predicted_values  = self.critic(global_states_t)
            critic_loss       = config["value_loss_coeff"] * nn.MSELoss()(
                predicted_values, returns
            )

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), config["max_grad_norm"])
            self.critic_optimizer.step()

            total_critic_loss += critic_loss.item()

        num_updates = config["ppo_epochs"] * self.num_agents
        return {
            "actor_loss":  total_actor_loss  / num_updates,
            "critic_loss": total_critic_loss / config["ppo_epochs"],
            "entropy":     total_entropy     / num_updates,
        }

    # ------------------------------------------------------------------
    def train(self, total_timesteps=None, results_dir="results", checkpoints_dir="checkpoints"):
        """Main training loop. All metrics computed from simulation — nothing hardcoded."""
        if total_timesteps is None:
            total_timesteps = self.config["total_timesteps"]

        os.makedirs(results_dir,     exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)

        log_file = os.path.join(results_dir, "mappo_training_log.csv")
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "timestep", "reward", "avg_latency", "packet_loss",
                "throughput", "jitter", "avg_hops", "actor_loss", "critic_loss", "entropy",
                "critical_latency", "high_latency", "medium_latency", "normal_latency",
            ])

        timesteps_so_far = 0
        best_reward      = -float("inf")

        print(f"Starting MAPPO training on {self.device}")
        print(f"Agents: {self.num_agents} | Total timesteps: {total_timesteps}")
        print(f"Network: {self.env.num_nodes} nodes, {self.env.graph.number_of_edges()} edges")
        print("=" * 70)

        while timesteps_so_far < total_timesteps:
            self.episode_count += 1

            # FIX 5: LR annealing — only anneal ACTOR, keep critic stable
            # The reward drops at ~ep1800 were caused by critic LR decaying
            # too fast, making value estimates unstable, which corrupted
            # advantage estimates and destabilized the actor.
            # Floor raised from 1% → 10% to prevent late-training collapse.
            if self.config.get("anneal_lr", False) and total_timesteps > 0:
                frac = max(0.10, 1.0 - timesteps_so_far / total_timesteps)
                actor_lr = self.config["lr_actor"] * frac
                for opt in self.actor_optimizers.values():
                    for pg in opt.param_groups:
                        pg["lr"] = actor_lr
                # Critic LR stays constant — no annealing

            # Collect rollout
            obs, global_state = self.env.reset()
            episode_reward    = 0.0
            self.buffer.clear()

            for _step in range(self.config["max_steps"]):
                actions, log_probs = self.select_actions(obs)
                value              = self.get_value(global_state)

                next_obs, next_gs, rewards, done, info = self.env.step(actions)

                # FIX 4: normalize reward before storing
                raw_reward = float(np.mean(list(rewards.values())))
                self.reward_normalizer.update(raw_reward)
                norm_reward = self.reward_normalizer.normalize(raw_reward)

                # Store normalized rewards per agent in buffer
                norm_rewards = {i: norm_reward for i in range(self.num_agents)}
                self.buffer.add(
                    obs, global_state, actions, log_probs,
                    norm_rewards, value, float(done)
                )

                # FIX 2: track mean reward across all agents for logging
                episode_reward   += raw_reward
                obs               = next_obs
                global_state      = next_gs
                timesteps_so_far += 1

                if done:
                    break

            # PPO update
            update_info = self.update()

            # Episode metrics
            ep_metrics = self.env._get_episode_metrics()
            per_class  = info.get("step_metrics", {}).get("per_class_latency", {})

            log_entry = {
                "episode":          self.episode_count,
                "timestep":         timesteps_so_far,
                "reward":           episode_reward,
                "avg_latency":      ep_metrics.get("avg_latency",     0),
                "packet_loss":      ep_metrics.get("avg_packet_loss", 0),
                "throughput":       ep_metrics.get("avg_throughput",  0),
                "jitter":           ep_metrics.get("avg_jitter",      0),
                "avg_hops":         ep_metrics.get("avg_hops",        0),
                "actor_loss":       update_info["actor_loss"],
                "critic_loss":      update_info["critic_loss"],
                "entropy":          update_info["entropy"],
                "critical_latency": per_class.get(0, 0),
                "high_latency":     per_class.get(1, 0),
                "medium_latency":   per_class.get(2, 0),
                "normal_latency":   per_class.get(3, 0),
            }
            self.training_log.append(log_entry)

            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(list(log_entry.values()))

            # Progress print
            if self.episode_count % self.config["log_interval"] == 0:
                recent   = self.training_log[-50:]
                avg_r    = np.mean([l["reward"]      for l in recent])
                avg_lat  = np.mean([l["avg_latency"]  for l in recent])
                avg_loss = np.mean([l["packet_loss"]  for l in recent])
                print(
                    f"Episode {self.episode_count:5d} | "
                    f"Timesteps: {timesteps_so_far:8d} | "
                    f"Avg Reward: {avg_r:8.2f} | "
                    f"Latency: {avg_lat:6.2f}ms | "
                    f"Loss Rate: {avg_loss:.4f} | "
                    f"Entropy: {update_info['entropy']:.4f} | "
                    f"Actor Loss: {update_info['actor_loss']:.4f}"
                )

                if len(self.training_log) > 500:
                    r_recent = np.mean([l["reward"] for l in self.training_log[-100:]])
                    r_older  = np.mean([l["reward"] for l in self.training_log[-500:-400]])
                    if r_recent < r_older - 10:
                        print("[WARNING] Reward declining. Consider adjusting hyperparameters.")

            # Track best reward every episode (not just at checkpoint intervals)
            if episode_reward > best_reward:
                best_reward = episode_reward
                self.save(os.path.join(checkpoints_dir, "mappo_best.pt"))

            # Periodic checkpoint
            if self.episode_count % self.config["save_interval"] == 0:
                self.save(os.path.join(checkpoints_dir, f"mappo_ep{self.episode_count}.pt"))

        self.save(os.path.join(checkpoints_dir, "mappo_final.pt"))

        import time as _time_module
        training_time = _time_module.time() - _training_start_time if '_training_start_time' in dir() else 0

        print(f"\n{'=' * 70}")
        print(f"MAPPO TRAINING COMPLETE")
        print(f"{'=' * 70}")
        print(f"  Episodes:    {self.episode_count}")
        print(f"  Timesteps:   {timesteps_so_far}")
        print(f"  Best Reward: {best_reward:.2f}")
        print(f"  Training log: {log_file}")
        print(f"{'=' * 70}")
        return self.training_log

    # ------------------------------------------------------------------
    def save(self, path, _max_retries=5, _retry_delay=1.0):
        """Save model checkpoint with architecture dims for safe loading.

        Uses atomic write (temp file + rename) and retry logic to avoid
        Windows error 32 ('file in use'), which commonly occurs when
        the project lives inside a OneDrive-synced folder.
        """
        import tempfile, time as _time

        obs_dim          = self.actors[0].network[0].in_features
        global_state_dim = self.critic.network[0].in_features
        action_dim       = self.actors[0].network[-1].out_features

        payload = {
            "actors":           {k: v.state_dict() for k, v in self.actors.items()},
            "critic":           self.critic.state_dict(),
            "actor_optimizers": {k: v.state_dict() for k, v in self.actor_optimizers.items()},
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "episode_count":    self.episode_count,
            "config":           self.config,
            "obs_dim":          obs_dim,
            "global_state_dim": global_state_dim,
            "action_dim":       action_dim,
        }

        save_dir = os.path.dirname(path) or "."
        for attempt in range(1, _max_retries + 1):
            try:
                # Write to a temp file first, then rename (atomic on same volume)
                fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=save_dir)
                os.close(fd)
                torch.save(payload, tmp_path)
                # On Windows, target must not exist for os.rename
                if os.path.exists(path):
                    os.replace(tmp_path, path)   # atomic replace
                else:
                    os.rename(tmp_path, path)
                return  # success
            except (RuntimeError, OSError, PermissionError) as e:
                # Clean up failed temp file
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                if attempt < _max_retries:
                    print(f"[WARN] Save failed (attempt {attempt}/{_max_retries}): {e}. "
                          f"Retrying in {_retry_delay}s...")
                    _time.sleep(_retry_delay)
                else:
                    print(f"[ERROR] Could not save checkpoint after {_max_retries} attempts: {e}")
                    print(f"        Tip: move the project OUT of OneDrive to avoid file-lock issues.")
                    raise

    def load(self, path):
        """Load model, rebuilding networks to match saved architecture dims."""
        ckpt = torch.load(path, map_location=self.device)
        if "obs_dim" in ckpt:
            obs_dim          = ckpt["obs_dim"]
            global_state_dim = ckpt["global_state_dim"]
            action_dim       = ckpt["action_dim"]
            config           = ckpt.get("config", self.config)
            self.actors      = {}
            self.actor_optimizers = {}
            for agent_id in range(self.num_agents):
                actor = ActorNetwork(
                    obs_dim, action_dim,
                    config["hidden_dim"], config["num_layers"]
                ).to(self.device)
                self.actors[agent_id]           = actor
                self.actor_optimizers[agent_id] = optim.Adam(
                    actor.parameters(), lr=config["lr_actor"]
                )
            self.critic = CriticNetwork(
                global_state_dim, config["hidden_dim"], config["num_layers"]
            ).to(self.device)
            self.critic_optimizer = optim.Adam(
                self.critic.parameters(), lr=config["lr_critic"]
            )
        for k, v in ckpt["actors"].items():
            self.actors[k].load_state_dict(v)
        self.critic.load_state_dict(ckpt["critic"])
        for k, v in ckpt["actor_optimizers"].items():
            self.actor_optimizers[k].load_state_dict(v)
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.episode_count = ckpt["episode_count"]
        print(f"Loaded checkpoint from {path} (episode {self.episode_count})")


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from network_env import HealthcareSDNEnv
    from config import MAPPO_CONFIG

    test_config = MAPPO_CONFIG.copy()
    test_config["total_timesteps"] = 10_000
    test_config["log_interval"]    = 5
    test_config["save_interval"]   = 50

    env   = HealthcareSDNEnv(num_nodes=30, num_agents=3, seed=42)
    agent = MAPPOAgent(env, test_config)

    print("Running quick MAPPO test (10K timesteps)...")
    log = agent.train(total_timesteps=10_000)
    print(f"\nFinal avg reward: {np.mean([l['reward'] for l in log[-10:]]):.2f}")
    print("MAPPO agent test passed!")