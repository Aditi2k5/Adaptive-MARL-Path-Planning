"""
rl_baselines.py
Implements DQN, IQL, VDN, PPO (single-agent), MAPPO for route selection.
All share the same environment and state/action space as QMIX.
Run: python rl_baselines.py --city bangalore --episodes 3000
"""
 
import os, random, argparse
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import config
from agent import build_agent_state, build_global_state, QNet
from environment import QuickCommerceEnv
from road_network import RoadNetwork
from qmix import MixingNetwork, ReplayBuffer


def _require_cuda(device: torch.device = None) -> torch.device:
    if device is not None:
        return device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. This project is configured to run on CUDA only.")
    return torch.device("cuda")
 
# ── DQN (single shared Q-network for all riders) ─────────────────────────────
class DQNAgent:
    """
    Single DQN agent shared across all riders.
    No agent-specific networks — treats all riders as one.
    Baseline to show multi-agent is needed.
    """
    def __init__(self, device: torch.device = None):
        self.device = _require_cuda(device)
        self.q_net  = QNet().to(self.device)
        self.t_net  = QNet().to(self.device)
        self.t_net.load_state_dict(self.q_net.state_dict())
        self.opt    = optim.Adam(self.q_net.parameters(), lr=config.LEARNING_RATE)
        self.buf    = deque(maxlen=config.REPLAY_BUFFER_SIZE)
        self.epsilon= config.EPSILON_START
 
    def act(self, obs, epsilon=None):
        eps = epsilon if epsilon is not None else self.epsilon
        if random.random() < eps:
            return random.randint(0, config.ACTION_DIM - 1)
        with torch.no_grad():
            obs_arr = np.asarray(obs, dtype=np.float32)
            return int(self.q_net(torch.from_numpy(obs_arr).unsqueeze(0).to(self.device))[0].argmax())
 
    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, float(done)))
 
    def train_step(self):
        if len(self.buf) < config.BATCH_SIZE: return None
        batch = random.sample(self.buf, config.BATCH_SIZE)
        s,a,r,s2,d = zip(*batch)
        s_t  = torch.FloatTensor(np.stack(s)).to(self.device)
        a_t  = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r_t  = torch.FloatTensor(r).to(self.device)
        s2_t = torch.FloatTensor(np.stack(s2)).to(self.device)
        d_t  = torch.FloatTensor(d).to(self.device)
        q    = self.q_net(s_t).gather(1, a_t).squeeze()
        with torch.no_grad():
            q2 = self.t_net(s2_t).max(1)[0]
            tg = r_t + config.GAMMA * q2 * (1 - d_t)
        loss = nn.MSELoss()(q, tg)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)
        self.opt.step(); return loss.item()
 
    def sync_target(self):
        self.t_net.load_state_dict(self.q_net.state_dict())
 
    def decay_epsilon(self):
        self.epsilon = max(config.EPSILON_END, self.epsilon * config.EPSILON_DECAY)
 
 
# ── IQL (Independent Q-Learning: N separate Q-networks, no coordination) ─────
class IQLCoordinator:
    """
    Independent Q-Learning: each rider has its own Q-network
    but they are trained independently with no mixing network.
    No global state used. Shows value of QMIX coordination.
    """
    def __init__(self, device: torch.device = None):
        self.device = _require_cuda(device)
        self.agents = [DQNAgent(device=self.device) for _ in range(config.N_RIDERS)]
 
    def act(self, obs_list, epsilon=None):
        return [self.agents[i].act(obs_list[i], epsilon) for i in range(config.N_RIDERS)]
 
    def push(self, obs_list, acts, rewards_list, nobs_list, done):
        for i in range(config.N_RIDERS):
            self.agents[i].push(obs_list[i], acts[i], rewards_list, nobs_list[i], done)
 
    def train(self):
        losses = [a.train_step() for a in self.agents]
        return np.mean([l for l in losses if l is not None]) if any(l is not None for l in losses) else None
 
    def sync_targets(self):
        for a in self.agents: a.sync_target()
 
    def decay_epsilon(self):
        for a in self.agents: a.decay_epsilon()
 
 
# ── VDN (Value Decomposition Networks: additive mixing, no hypernetwork) ──────
class VDNMixer(nn.Module):
    """
    VDN: Q_tot = sum of individual Q_j values.
    No monotonicity via hypernetwork — just addition.
    Simpler than QMIX. Shows value of QMIX's state-conditional mixing.
    """
    def forward(self, agent_qs, state=None):
        return agent_qs.sum(dim=1)   # simple sum, no state conditioning
 
 
class VDNCoordinator:
    def __init__(self, device: torch.device = None):
        self.device = _require_cuda(device)
        self.agents  = [DQNAgent(device=self.device) for _ in range(config.N_RIDERS)]
        self.mixer   = VDNMixer().to(self.device)
        self.t_mixer = VDNMixer().to(self.device)
        params = []
        for a in self.agents: params += list(a.q_net.parameters())
        self.opt = optim.Adam(params, lr=config.LEARNING_RATE)
        self.buf = ReplayBuffer()
        self.epsilon = config.EPSILON_START
 
    def act(self, obs_list, epsilon=None):
        eps = epsilon if epsilon is not None else self.epsilon
        return [self.agents[i].act(obs_list[i], eps) for i in range(config.N_RIDERS)]
 
    def push_transition(self, gs, obs, acts, reward, ngs, nobs, done, weather=0):
        self.buf.push((gs, obs, acts, float(reward), ngs, nobs, float(done)), weather)
 
    def train(self):
        if len(self.buf) < config.BATCH_SIZE: return None
        gs, obs, acts, rews, ngs, nobs, dones = self.buf.sample(config.BATCH_SIZE)
        obs_t  = torch.FloatTensor(obs).to(self.device)
        acts_t = torch.LongTensor(acts).to(self.device)
        rews_t = torch.FloatTensor(rews).to(self.device)
        nobs_t = torch.FloatTensor(nobs).to(self.device)
        done_t = torch.FloatTensor(dones).to(self.device)
        cur_qs = torch.cat([self.agents[i].q_net(obs_t[:,i,:]).gather(
                    1, acts_t[:,i].unsqueeze(1)) for i in range(config.N_RIDERS)], dim=1)
        q_tot  = self.mixer(cur_qs)
        with torch.no_grad():
            tgt_qs = torch.cat([self.agents[i].t_net(nobs_t[:,i,:]).max(1)[0].unsqueeze(1)
                                for i in range(config.N_RIDERS)], dim=1)
            targets = rews_t + config.GAMMA * self.t_mixer(tgt_qs) * (1 - done_t)
        loss = nn.MSELoss()(q_tot, targets)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(sum([list(a.q_net.parameters()) for a in self.agents], []), 10)
        self.opt.step(); return loss.item()
 
    def sync_targets(self):
        for a in self.agents: a.sync_target()
 
    def decay_epsilon(self):
        self.epsilon = max(config.EPSILON_END, self.epsilon * config.EPSILON_DECAY)
 
 
# ── PPO Actor-Critic (single shared policy) ───────────────────────────────────
class PPONet(nn.Module):
    def __init__(self, device: torch.device = None):
        super().__init__()
        self.device = _require_cuda(device)
        self.shared = nn.Sequential(nn.Linear(config.STATE_DIM, 128), nn.ReLU(),
                                    nn.Linear(128, 128), nn.ReLU())
        self.actor  = nn.Linear(128, config.ACTION_DIM)
        self.critic = nn.Linear(128, 1)
        self.to(self.device)
 
    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)
 
    def act(self, obs):
        with torch.no_grad():
            logits, val = self.forward(torch.FloatTensor(obs).unsqueeze(0).to(self.device))
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            return a.item(), dist.log_prob(a).item(), val.item()
 
 
class PPOAgent:
    def __init__(self, device: torch.device = None):
        self.device = _require_cuda(device)
        self.net = PPONet(device=self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=3e-4)
        self.buf = []   # (obs, act, logp, val, rew, done)
        self.epsilon = 0.0   # PPO doesn't use epsilon
 
    def act(self, obs, epsilon=None):
        a, _, _ = self.net.act(obs); return a
 
    def push(self, obs, act, logp, val, rew, done):
        self.buf.append((obs, act, logp, val, rew, done))
 
    def train(self, gamma=0.95, clip=0.2, epochs=4):
        if len(self.buf) < 64: return None
        obs,acts,logps,vals,rews,dones = zip(*self.buf)
        obs_t  = torch.FloatTensor(np.stack(obs)).to(self.device)
        acts_t = torch.LongTensor(acts).to(self.device)
        logp_t = torch.FloatTensor(logps).to(self.device)
        # Compute returns
        returns = []; G = 0
        for r, d in zip(reversed(rews), reversed(dones)):
            G = r + gamma * G * (1 - d); returns.insert(0, G)
        ret_t = torch.FloatTensor(returns).to(self.device)
        adv_t = ret_t - torch.FloatTensor(vals).to(self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        total_loss = 0
        for _ in range(epochs):
            logits, val = self.net(obs_t)
            dist   = torch.distributions.Categorical(logits=logits)
            new_lp = dist.log_prob(acts_t)
            ratio  = (new_lp - logp_t).exp()
            s1 = ratio * adv_t
            s2 = torch.clamp(ratio, 1-clip, 1+clip) * adv_t
            actor_loss  = -torch.min(s1, s2).mean()
            critic_loss = nn.MSELoss()(val.squeeze(), ret_t)
            loss = actor_loss + 0.5 * critic_loss
            self.opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.opt.step(); total_loss += loss.item()
        self.buf.clear()
        return total_loss / epochs
 
 
# ── MAPPO (Multi-Agent PPO with shared critic on global state) ────────────────
class MAPPOCoordinator:
    def __init__(self, device: torch.device = None):
        self.device = _require_cuda(device)
        self.agents = [PPOAgent(device=self.device) for _ in range(config.N_RIDERS)]
        # Shared centralised critic
        self.critic = nn.Sequential(
            nn.Linear(config.GLOBAL_STATE_DIM, 128), nn.ReLU(),
            nn.Linear(128, 1))
        self.critic = self.critic.to(self.device)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=3e-4)
        self.epsilon = 0.0
 
    def act(self, obs_list, epsilon=None):
        return [self.agents[i].act(obs_list[i]) for i in range(config.N_RIDERS)]
 
    def train(self):
        losses = [a.train(epochs=3) for a in self.agents]
        return np.mean([l for l in losses if l is not None]) if any(l is not None for l in losses) else None
 
    def decay_epsilon(self): pass   # PPO-based, no epsilon
    def sync_targets(self): pass