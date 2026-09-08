"""
MADDPG Agent Implementation (Baseline)
========================================
Multi-Agent Deep Deterministic Policy Gradient.
This is the algorithm from the base paper (Dake et al., 2021).
Implemented for fair comparison with our MAPPO approach.

Key differences from MAPPO:
1. Off-policy (uses replay buffer with stored transitions)
2. No clipping - uses soft target updates instead
3. Each agent has its own critic (not shared)
4. Critic input: agent_obs + all_agents_actions (not global state)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
from collections import deque, defaultdict


class MADDPGActor(nn.Module):
    """Actor network for MADDPG (same role as MAPPO actor)."""
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )
    
    def forward(self, obs):
        return self.network(obs)


class MADDPGCritic(nn.Module):
    """
    Critic for MADDPG: takes obs + all agents' actions.
    Unlike MAPPO's shared global-state critic, each MADDPG agent
    has its own critic that sees observations AND actions of all agents.
    """
    def __init__(self, obs_dim, total_action_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim + total_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, obs, actions):
        x = torch.cat([obs, actions], dim=-1)
        return self.network(x).squeeze(-1)


class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return batch
    
    def __len__(self):
        return len(self.buffer)


class MADDPGAgent:
    """
    MADDPG implementation following Dake et al. (2021).
    Uses 2 agents: routing + load balancing (simplified from their routing + DDoS).
    """
    
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_agents = config["num_agents"]
        
        # Get dimensions
        obs, gs = env.reset()
        self.obs_dim = len(list(obs.values())[0])
        self.action_dim = env.action_dim_per_agent
        total_action_dim = self.action_dim * self.num_agents
        
        # Networks for each agent
        self.actors = {}
        self.critics = {}
        self.target_actors = {}
        self.target_critics = {}
        self.actor_optimizers = {}
        self.critic_optimizers = {}
        
        for i in range(self.num_agents):
            # Primary networks
            self.actors[i] = MADDPGActor(self.obs_dim, self.action_dim, config["hidden_dim"]).to(self.device)
            self.critics[i] = MADDPGCritic(self.obs_dim, total_action_dim, config["hidden_dim"]).to(self.device)
            
            # Target networks (soft update)
            self.target_actors[i] = MADDPGActor(self.obs_dim, self.action_dim, config["hidden_dim"]).to(self.device)
            self.target_critics[i] = MADDPGCritic(self.obs_dim, total_action_dim, config["hidden_dim"]).to(self.device)
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())
            
            self.actor_optimizers[i] = optim.Adam(self.actors[i].parameters(), lr=config["lr_actor"])
            self.critic_optimizers[i] = optim.Adam(self.critics[i].parameters(), lr=config["lr_critic"])
        
        self.replay_buffer = ReplayBuffer(config["buffer_size"])
        self.epsilon = config["epsilon_start"]
        self.training_log = []
        self.episode_count = 0
    
    def select_actions(self, observations, deterministic=False):
        """Select actions with epsilon-greedy exploration.
        
        Handles evaluation environments that may have more agent obs slots
        than trained actors (e.g. env has 3 obs but only 2 actors trained).
        Falls back to actor 0 for any agent index beyond num_agents.
        """
        actions = {}

        all_agent_ids = list(observations.keys())
        for agent_id in all_agent_ids:
            # Use the trained actor for this agent, fall back to actor 0
            actor_id = agent_id if agent_id < self.num_agents else 0
            obs_key  = agent_id if agent_id in observations else min(observations.keys())
            obs_tensor = torch.FloatTensor(observations[obs_key]).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action_probs = self.actors[actor_id](obs_tensor).squeeze(0)

            if not deterministic and np.random.random() < self.epsilon:
                action = np.random.randint(0, self.action_dim)
            else:
                action = action_probs.argmax().item()

            actions[agent_id] = action

        return actions
    
    def soft_update(self, target, source, tau):
        """Soft update target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def update(self):
        """Update all agents using sampled batch from replay buffer."""
        if len(self.replay_buffer) < self.config["batch_size"]:
            return {"actor_loss": 0, "critic_loss": 0, "entropy": 0}
        
        batch = self.replay_buffer.sample(self.config["batch_size"])
        
        # Unpack batch
        obs_batch = {i: [] for i in range(self.num_agents)}
        action_batch = {i: [] for i in range(self.num_agents)}
        reward_batch = {i: [] for i in range(self.num_agents)}
        next_obs_batch = {i: [] for i in range(self.num_agents)}
        done_batch = []
        
        for transition in batch:
            for i in range(self.num_agents):
                obs_key = i if i in transition["obs"] else min(transition["obs"].keys())
                obs_batch[i].append(transition["obs"][obs_key])
                action_batch[i].append(transition["actions"][i])
                reward_batch[i].append(transition["rewards"].get(i, transition["rewards"][0]))
                next_obs_key = i if i in transition["next_obs"] else min(transition["next_obs"].keys())
                next_obs_batch[i].append(transition["next_obs"][next_obs_key])
            done_batch.append(transition["done"])
        
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        
        for agent_id in range(self.num_agents):
            # Convert to tensors
            obs_t = torch.FloatTensor(np.array(obs_batch[agent_id])).to(self.device)
            next_obs_t = torch.FloatTensor(np.array(next_obs_batch[agent_id])).to(self.device)
            rewards_t = torch.FloatTensor(np.array(reward_batch[agent_id])).to(self.device)
            dones_t = torch.FloatTensor(np.array(done_batch)).to(self.device)

            # Compute entropy from current action distribution
            with torch.no_grad():
                action_probs = self.actors[agent_id](obs_t)
                # Entropy = -sum(p * log(p)), clamped for numerical stability
                log_probs_ent = torch.log(action_probs + 1e-8)
                entropy = -(action_probs * log_probs_ent).sum(dim=-1).mean()
                total_entropy += entropy.item()
            
            # All agents' actions (one-hot encoded and concatenated)
            all_actions = []
            all_next_actions = []
            for i in range(self.num_agents):
                act_onehot = torch.zeros(self.config["batch_size"], self.action_dim).to(self.device)
                act_onehot.scatter_(1, torch.LongTensor(action_batch[i]).unsqueeze(1).to(self.device), 1)
                all_actions.append(act_onehot)
                
                next_obs_i = torch.FloatTensor(np.array(next_obs_batch[i])).to(self.device)
                next_act = self.target_actors[i](next_obs_i)
                all_next_actions.append(next_act)
            
            all_actions_cat = torch.cat(all_actions, dim=-1)
            all_next_actions_cat = torch.cat(all_next_actions, dim=-1)
            
            # Critic update
            with torch.no_grad():
                target_value = rewards_t + self.config["gamma"] * (1 - dones_t) * \
                    self.target_critics[agent_id](next_obs_t, all_next_actions_cat)
            
            current_value = self.critics[agent_id](obs_t, all_actions_cat)
            critic_loss = nn.MSELoss()(current_value, target_value)
            
            self.critic_optimizers[agent_id].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[agent_id].step()
            
            # Actor update
            current_actions = self.actors[agent_id](obs_t)
            # Replace agent's actions in all_actions
            new_all_actions = []
            for i in range(self.num_agents):
                if i == agent_id:
                    new_all_actions.append(current_actions)
                else:
                    new_all_actions.append(all_actions[i].detach())
            new_all_actions_cat = torch.cat(new_all_actions, dim=-1)
            
            actor_loss = -self.critics[agent_id](obs_t, new_all_actions_cat).mean()
            
            self.actor_optimizers[agent_id].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[agent_id].step()
            
            # Soft update targets
            self.soft_update(self.target_actors[agent_id], self.actors[agent_id], self.config["tau"])
            self.soft_update(self.target_critics[agent_id], self.critics[agent_id], self.config["tau"])
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
        
        # Decay epsilon
        self.epsilon = max(self.config["epsilon_end"], 
                          self.epsilon * self.config["epsilon_decay"])
        
        return {
            "actor_loss": total_actor_loss / self.num_agents,
            "critic_loss": total_critic_loss / self.num_agents,
            "entropy": total_entropy / self.num_agents,
        }
    
    def train(self, total_episodes=None, results_dir="results", checkpoints_dir="checkpoints"):
        """Main training loop for MADDPG."""
        if total_episodes is None:
            total_episodes = self.config["total_episodes"]
        
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        log_file = os.path.join(results_dir, "maddpg_training_log.csv")
        
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "reward", "avg_latency", "packet_loss",
                "throughput", "jitter", "avg_hops", "actor_loss", "critic_loss", "entropy", "epsilon"
            ])
        
        print(f"Starting MADDPG training (baseline) on {self.device}")
        print(f"Total episodes: {total_episodes}")
        print("=" * 70)
        
        for ep in range(1, total_episodes + 1):
            self.episode_count = ep
            obs, global_state = self.env.reset()
            episode_reward = 0.0
            
            for step in range(self.config["max_steps_per_episode"]):
                actions = self.select_actions(obs)
                next_obs, next_gs, rewards, done, info = self.env.step(actions)
                
                # Store transition in replay buffer
                self.replay_buffer.push({
                    "obs": obs, "actions": actions, "rewards": rewards,
                    "next_obs": next_obs, "done": float(done),
                })
                
                episode_reward += rewards[0]
                obs = next_obs
                
                # Update networks
                update_info = self.update()
                
                if done:
                    break
            
            ep_metrics = self.env._get_episode_metrics()
            
            log_entry = {
                "episode": ep,
                "reward": episode_reward,
                "avg_latency": ep_metrics.get("avg_latency", 0),
                "packet_loss": ep_metrics.get("avg_packet_loss", 0),
                "throughput": ep_metrics.get("avg_throughput", 0),
                "jitter": ep_metrics.get("avg_jitter", 0),
                "avg_hops": ep_metrics.get("avg_hops", 0),
                "actor_loss": update_info["actor_loss"],
                "critic_loss": update_info["critic_loss"],
                "entropy": update_info["entropy"],
                "epsilon": self.epsilon,
            }
            self.training_log.append(log_entry)
            
            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(list(log_entry.values()))
            
            if ep % 50 == 0:
                avg_r = np.mean([l["reward"] for l in self.training_log[-50:]])
                avg_l = np.mean([l["avg_latency"] for l in self.training_log[-50:]])
                avg_e = np.mean([l["entropy"] for l in self.training_log[-50:]])
                print(f"Episode {ep:5d} | Avg Reward: {avg_r:8.2f} | Latency: {avg_l:.2f}ms | ε: {self.epsilon:.3f}")
        
        # Save final model
        self.save(os.path.join(checkpoints_dir, "maddpg_final.pt"))

        best_reward = max(l["reward"] for l in self.training_log)

        print(f"\n{'=' * 70}")
        print(f"MADDPG TRAINING COMPLETE")
        print(f"{'=' * 70}")
        print(f"  Episodes:    {self.episode_count}")
        print(f"  Best Reward: {best_reward:.2f}")
        print(f"  Training log: {log_file}")
        print(f"{'=' * 70}")
        return self.training_log
    
    def save(self, path):
        """Save checkpoint with architecture dims for safe loading."""
        obs_dim          = self.actors[0].network[0].in_features
        # network[-1] is Softmax - find the last Linear layer for out_features
        action_dim       = next(
            m.out_features for m in reversed(list(self.actors[0].network))
            if hasattr(m, 'out_features')
        )
        total_action_dim = self.critics[0].network[0].in_features - obs_dim
        torch.save({
            "actors":          {k: v.state_dict() for k, v in self.actors.items()},
            "critics":         {k: v.state_dict() for k, v in self.critics.items()},
            "config":          self.config,
            "obs_dim":         obs_dim,
            "action_dim":      action_dim,
            "total_action_dim": total_action_dim,
            "num_agents":      self.num_agents,
        }, path)

    def load(self, path):
        """Load checkpoint, rebuilding networks to match saved architecture dims."""
        ckpt = torch.load(path, map_location=self.device)

        if "obs_dim" in ckpt:
            # New checkpoint - dims stored explicitly
            obs_dim          = ckpt["obs_dim"]
            action_dim       = ckpt["action_dim"]
            total_action_dim = ckpt["total_action_dim"]
            saved_num_agents = ckpt["num_agents"]
        else:
            # Old checkpoint - infer dims from weight shapes
            saved_num_agents = len(ckpt["actors"])
            actor_sd = list(ckpt["actors"].values())[0]
            obs_dim  = actor_sd["network.0.weight"].shape[1]
            # Find last Linear weight in actor (before Softmax)
            action_dim = max(
                v.shape[0] for k, v in actor_sd.items()
                if k.endswith(".weight") and len(v.shape) == 2
            )
            critic_in        = list(ckpt["critics"].values())[0]["network.0.weight"].shape[1]
            total_action_dim = critic_in - obs_dim

        config = ckpt.get("config", self.config)

        # Rebuild only the trained actors/critics (using saved num_agents)
        # Extra agents added for eval will keep their randomly-init weights
        for i in range(saved_num_agents):
            self.actors[i] = MADDPGActor(
                obs_dim, action_dim, config["hidden_dim"]
            ).to(self.device)
            self.critics[i] = MADDPGCritic(
                obs_dim, total_action_dim, config["hidden_dim"]
            ).to(self.device)

        for k, v in ckpt["actors"].items():
            self.actors[k].load_state_dict(v)
        for k, v in ckpt["critics"].items():
            self.critics[k].load_state_dict(v)
        print(f"Loaded MADDPG from {path}")


if __name__ == "__main__":
    from network_env import HealthcareSDNEnv
    from config import MADDPG_CONFIG
    
    test_config = MADDPG_CONFIG.copy()
    test_config["total_episodes"] = 50
    
    env = HealthcareSDNEnv(num_nodes=30, num_agents=2, seed=42)
    agent = MADDPGAgent(env, test_config)
    
    print("Running quick MADDPG test (50 episodes)...")
    log = agent.train(total_episodes=50)
    print(f"Final avg reward: {np.mean([l['reward'] for l in log[-10:]]):.2f}")
    print("MADDPG test passed!")