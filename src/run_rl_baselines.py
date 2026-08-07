"""
run_rl_baselines.py — Complete corrected version (v3)

FIX vs v2: DQN, IQL, and VDN now use the SAME flood-prioritised replay
scheme as QMIX (30% of each batch drawn from heavy-rain transitions when
available). Previously only QMIX oversampled flood episodes while the
other buffer-based methods sampled uniformly — this asymmetry inflated
QMIX's training-batch variance relative to every baseline in any city
where the natural heavy-rain frequency is well below 30% (i.e. every
city except Mumbai), producing artificially high eval-time std and
worse J for QMIX that had nothing to do with its architecture.
"""

from __future__ import annotations
import argparse, logging, os, pickle, random
from collections import deque
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import config
from agent import build_agent_state, build_global_state
from environment import QuickCommerceEnv
from qmix import QMIXCoordinator
from road_network import RoadNetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOGGER = logging.getLogger("rl_baselines")

def _setup_logger(city: str) -> None:
    os.makedirs("results", exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    for h in list(LOGGER.handlers):
        LOGGER.removeHandler(h); h.close()
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    fh  = logging.FileHandler(f"results/rl_baselines_{city}.log", "w", "utf-8")
    sh  = logging.StreamHandler()
    for handler in [fh, sh]:
        handler.setFormatter(fmt)
        LOGGER.addHandler(handler)

def log(msg: str) -> None:
    LOGGER.info(msg)


# =============================================================================
# Shared prioritised replay buffer — IDENTICAL scheme for every buffer-based
# method (DQN, IQL, VDN, QMIX). MAPPO is on-policy and does not use this.
# =============================================================================

class PrioritizedReplayBuffer:
    """
    30% of each sampled minibatch drawn from heavy-rain (weather_state>=2)
    transitions when available, 70% uniform. Matches qmix.py's ReplayBuffer
    exactly so no method gets an unfair training-distribution advantage.
    """
    def __init__(self, capacity=config.REPLAY_BUFFER_SIZE, flood_capacity=5000):
        self.buf = deque(maxlen=capacity)
        self.flood_buf = deque(maxlen=flood_capacity)

    def push(self, item, weather_state=0):
        self.buf.append(item)
        if weather_state >= 2:
            self.flood_buf.append(item)

    def __len__(self):
        return len(self.buf)

    def sample(self, batch_size):
        n_flood = min(int(batch_size * 0.30), len(self.flood_buf))
        n_normal = batch_size - n_flood
        if n_flood > 0 and len(self.flood_buf) >= n_flood:
            flood_batch = random.sample(list(self.flood_buf), n_flood)
        else:
            flood_batch = []
            n_normal = batch_size
        normal_batch = random.sample(list(self.buf), n_normal)
        batch = flood_batch + normal_batch
        random.shuffle(batch)
        return batch


# =============================================================================
# Shared, agent-ID-conditioned Q-network
# =============================================================================

class SharedQNet(nn.Module):
    def __init__(self, obs_dim=config.STATE_DIM, n_actions=config.ACTION_DIM,
                n_agents=config.N_RIDERS, hidden=128):
        super().__init__()
        self.n_agents = n_agents
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_agents, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs, agent_id):
        one_hot = F.one_hot(agent_id, num_classes=self.n_agents).float()
        return self.net(torch.cat([obs, one_hot], dim=-1))


class SharedActorNet(nn.Module):
    def __init__(self, obs_dim=config.STATE_DIM, n_actions=config.ACTION_DIM,
                n_agents=config.N_RIDERS, hidden=128):
        super().__init__()
        self.n_agents = n_agents
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_agents, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs, agent_id):
        one_hot = F.one_hot(agent_id, num_classes=self.n_agents).float()
        return self.net(torch.cat([obs, one_hot], dim=-1))


def _agent_id_tensor(n: int, device) -> torch.Tensor:
    return torch.arange(n, dtype=torch.long, device=device)


# =============================================================================
# BASE COORDINATOR INTERFACE
# =============================================================================

class _BaseCoordinator:
    def act(self, obs_list, epsilon=0.0):
        raise NotImplementedError
    def push(self, gs, obs, acts, reward, ngs, nobs, done, weather=0):
        raise NotImplementedError
    def train_step(self):
        raise NotImplementedError
    def sync_targets(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# DQN — single shared net, no agent-id, NOW with flood-prioritised replay
# ─────────────────────────────────────────────────────────────────────────────

class DQNCoordinator(_BaseCoordinator):
    def __init__(self, device):
        self.device = device
        self.q_net  = self._make_net().to(device)
        self.t_net  = self._make_net().to(device)
        self.t_net.load_state_dict(self.q_net.state_dict())
        self.opt    = optim.Adam(self.q_net.parameters(), lr=config.LEARNING_RATE)
        self.buf    = PrioritizedReplayBuffer()   # ← FIX: was plain deque

    @staticmethod
    def _make_net():
        return nn.Sequential(
            nn.Linear(config.STATE_DIM, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, config.ACTION_DIM),
        )

    def act(self, obs_list, epsilon=0.0):
        obs_t = torch.FloatTensor(np.stack(obs_list)).to(self.device)
        with torch.no_grad():
            greedy = self.q_net(obs_t).argmax(dim=1).cpu().numpy()
        actions = []
        for i in range(len(obs_list)):
            if random.random() < epsilon:
                actions.append(random.randint(0, config.ACTION_DIM - 1))
            else:
                actions.append(int(greedy[i]))
        return actions

    def push(self, gs, obs, acts, reward, ngs, nobs, done, weather=0):
        for i in range(len(acts)):
            self.buf.push((obs[i], acts[i], reward, nobs[i], float(done)), weather)

    def train_step(self):
        if len(self.buf) < config.BATCH_SIZE:
            return None
        batch = self.buf.sample(config.BATCH_SIZE)   # ← FIX: prioritised sample
        s, a, r, s2, d = zip(*batch)
        s_t  = torch.FloatTensor(np.stack(s)).to(self.device)
        a_t  = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r_t  = torch.FloatTensor(r).to(self.device)
        s2_t = torch.FloatTensor(np.stack(s2)).to(self.device)
        d_t  = torch.FloatTensor(d).to(self.device)
        q    = self.q_net(s_t).gather(1, a_t).squeeze(1)
        with torch.no_grad():
            q2  = self.t_net(s2_t).max(1)[0]
            tgt = r_t + config.GAMMA * q2 * (1 - d_t)
        loss = nn.MSELoss()(q, tgt)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.opt.step()
        return loss.item()

    def sync_targets(self):
        self.t_net.load_state_dict(self.q_net.state_dict())


# ─────────────────────────────────────────────────────────────────────────────
# IQL — shared agent-id-conditioned weights, NOW with flood-prioritised replay
# ─────────────────────────────────────────────────────────────────────────────

class IQLCoordinator(_BaseCoordinator):
    def __init__(self, n_agents: int, device):
        self.n      = n_agents
        self.device = device
        self.q_net  = SharedQNet(n_agents=n_agents).to(device)
        self.t_net  = SharedQNet(n_agents=n_agents).to(device)
        self.t_net.load_state_dict(self.q_net.state_dict())
        self.opt    = optim.Adam(self.q_net.parameters(), lr=config.LEARNING_RATE)
        self.buf    = PrioritizedReplayBuffer()   # ← FIX: was plain deque
        self._agent_ids = _agent_id_tensor(n_agents, device)

    def act(self, obs_list, epsilon=0.0):
        obs_t = torch.FloatTensor(np.stack(obs_list)).to(self.device)
        with torch.no_grad():
            q = self.q_net(obs_t, self._agent_ids)
            greedy = q.argmax(dim=1).cpu().numpy()
        actions = []
        for i in range(self.n):
            if random.random() < epsilon:
                actions.append(random.randint(0, config.ACTION_DIM - 1))
            else:
                actions.append(int(greedy[i]))
        return actions

    def push(self, gs, obs, acts, reward, ngs, nobs, done, weather=0):
        for i in range(self.n):
            self.buf.push((obs[i], i, acts[i], reward, nobs[i], float(done)), weather)

    def train_step(self):
        if len(self.buf) < config.BATCH_SIZE:
            return None
        batch = self.buf.sample(config.BATCH_SIZE)   # ← FIX: prioritised sample
        s, aid, a, r, s2, d = zip(*batch)
        s_t   = torch.FloatTensor(np.stack(s)).to(self.device)
        aid_t = torch.LongTensor(aid).to(self.device)
        a_t   = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r_t   = torch.FloatTensor(r).to(self.device)
        s2_t  = torch.FloatTensor(np.stack(s2)).to(self.device)
        d_t   = torch.FloatTensor(d).to(self.device)
        q = self.q_net(s_t, aid_t).gather(1, a_t).squeeze(1)
        with torch.no_grad():
            q2  = self.t_net(s2_t, aid_t).max(1)[0]
            tgt = r_t + config.GAMMA * q2 * (1 - d_t)
        loss = nn.MSELoss()(q, tgt)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.opt.step()
        return loss.item()

    def sync_targets(self):
        self.t_net.load_state_dict(self.q_net.state_dict())


# ─────────────────────────────────────────────────────────────────────────────
# VDN — shared weights, additive Q_tot, NOW with flood-prioritised replay
# ─────────────────────────────────────────────────────────────────────────────

class VDNCoordinator(_BaseCoordinator):
    def __init__(self, n_agents: int, device, lr_scale: float = 0.5):
        self.n      = n_agents
        self.device = device
        self.q_net  = SharedQNet(n_agents=n_agents).to(device)
        self.t_net  = SharedQNet(n_agents=n_agents).to(device)
        self.t_net.load_state_dict(self.q_net.state_dict())
        self.opt    = optim.Adam(self.q_net.parameters(),
                                 lr=config.LEARNING_RATE * lr_scale)
        self.buf    = PrioritizedReplayBuffer()   # ← FIX: was plain deque
        self._agent_ids = _agent_id_tensor(n_agents, device)

    def act(self, obs_list, epsilon=0.0):
        obs_t = torch.FloatTensor(np.stack(obs_list)).to(self.device)
        with torch.no_grad():
            q = self.q_net(obs_t, self._agent_ids)
            greedy = q.argmax(dim=1).cpu().numpy()
        actions = []
        for i in range(self.n):
            if random.random() < epsilon:
                actions.append(random.randint(0, config.ACTION_DIM - 1))
            else:
                actions.append(int(greedy[i]))
        return actions

    def push(self, gs, obs, acts, reward, ngs, nobs, done, weather=0):
        # store the FULL joint transition; tag with weather for prioritisation
        self.buf.push(
            (np.stack(obs), np.array(acts), reward, np.stack(nobs), float(done)),
            weather
        )

    def train_step(self):
        if len(self.buf) < config.BATCH_SIZE:
            return None
        batch = self.buf.sample(config.BATCH_SIZE)   # ← FIX: prioritised sample
        obs, acts, rews, nobs, dones = zip(*batch)
        obs_t   = torch.FloatTensor(np.stack(obs)).to(self.device)
        acts_t  = torch.LongTensor(np.stack(acts)).to(self.device)
        rews_t  = torch.FloatTensor(rews).to(self.device)
        nobs_t  = torch.FloatTensor(np.stack(nobs)).to(self.device)
        done_t  = torch.FloatTensor(dones).to(self.device)

        B = obs_t.shape[0]
        flat_obs   = obs_t.reshape(B * self.n, -1)
        flat_aid   = self._agent_ids.repeat(B)
        flat_acts  = acts_t.reshape(B * self.n, 1)
        q_all      = self.q_net(flat_obs, flat_aid).gather(1, flat_acts).reshape(B, self.n)
        q_tot      = q_all.sum(dim=1)

        with torch.no_grad():
            flat_nobs = nobs_t.reshape(B * self.n, -1)
            q_next_all = self.t_net(flat_nobs, flat_aid).max(dim=1)[0].reshape(B, self.n)
            targets = rews_t + config.GAMMA * q_next_all.sum(dim=1) * (1 - done_t)

        loss = nn.MSELoss()(q_tot, targets)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.opt.step()
        return loss.item()

    def sync_targets(self):
        self.t_net.load_state_dict(self.q_net.state_dict())


# ─────────────────────────────────────────────────────────────────────────────
# MAPPO — unchanged (on-policy rollout, no replay buffer, so no confound here)
# ─────────────────────────────────────────────────────────────────────────────

class _CriticNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.GLOBAL_STATE_DIM, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MAPPOCoordinator(_BaseCoordinator):
    def __init__(self, n_agents: int, device,
                 clip: float = 0.2, ppo_epochs: int = 4, rollout_len: int = 256):
        self.n           = n_agents
        self.device      = device
        self.clip        = clip
        self.ppo_epochs  = ppo_epochs
        self.rollout_len = rollout_len

        self.actor  = SharedActorNet(n_agents=n_agents).to(device)
        self.critic = _CriticNet().to(device)
        self.actor_opt  = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=3e-4)
        self._agent_ids = _agent_id_tensor(n_agents, device)

        self._buf: list = []

    def act(self, obs_list, epsilon=0.0):
        obs_t = torch.FloatTensor(np.stack(obs_list)).to(self.device)
        with torch.no_grad():
            logits = self.actor(obs_t, self._agent_ids)
            dist   = torch.distributions.Categorical(logits=logits)
            a      = dist.sample()
        return a.cpu().numpy().tolist()

    def _log_probs(self, obs_t, acts_t):
        B = obs_t.shape[0]
        flat_obs  = obs_t.reshape(B * self.n, -1)
        flat_aid  = self._agent_ids.repeat(B)
        flat_acts = acts_t.reshape(B * self.n)
        logits    = self.actor(flat_obs, flat_aid)
        dist      = torch.distributions.Categorical(logits=logits)
        lp        = dist.log_prob(flat_acts).reshape(B, self.n)
        return lp

    def push(self, gs, obs, acts, reward, ngs, nobs, done, weather=0):
        self._buf.append({
            "gs": gs, "obs": np.array(obs),
            "acts": np.array(acts), "reward": float(reward),
            "ngs": ngs, "nobs": np.array(nobs),
            "done": float(done),
        })

    def train_step(self):
        if len(self._buf) < self.rollout_len:
            return None

        batch   = self._buf[-self.rollout_len:]
        gs_arr  = np.stack([b["gs"]   for b in batch])
        obs_arr = np.stack([b["obs"]  for b in batch])
        act_arr = np.stack([b["acts"] for b in batch])
        rew_arr = np.array([b["reward"] for b in batch], dtype=np.float32)
        ngs_arr = np.stack([b["ngs"]  for b in batch])
        done_arr= np.array([b["done"] for b in batch], dtype=np.float32)

        gs_t   = torch.FloatTensor(gs_arr).to(self.device)
        obs_t  = torch.FloatTensor(obs_arr).to(self.device)
        act_t  = torch.LongTensor(act_arr).to(self.device)
        rew_t  = torch.FloatTensor(rew_arr).to(self.device)
        ngs_t  = torch.FloatTensor(ngs_arr).to(self.device)
        done_t = torch.FloatTensor(done_arr).to(self.device)

        with torch.no_grad():
            val_next = self.critic(ngs_t)
            returns  = rew_t + config.GAMMA * val_next * (1 - done_t)
            vals_old = self.critic(gs_t)
            adv      = returns - vals_old
            adv      = (adv - adv.mean()) / (adv.std() + 1e-8)
            old_lp   = self._log_probs(obs_t, act_t)

        total_loss = 0.0
        for _ in range(self.ppo_epochs):
            new_lp = self._log_probs(obs_t, act_t)
            ratio  = (new_lp - old_lp.detach()).exp().mean(dim=1)
            s1     = ratio * adv
            s2     = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv
            actor_loss = -torch.min(s1, s2).mean()

            vals_new    = self.critic(gs_t)
            critic_loss = nn.MSELoss()(vals_new, returns)

            loss = actor_loss + 0.5 * critic_loss
            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.actor_opt.step()
            self.critic_opt.step()
            total_loss += loss.item()

        self._buf.clear()
        return total_loss / self.ppo_epochs

    def sync_targets(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# QMIX wrapper — unchanged
# ─────────────────────────────────────────────────────────────────────────────

class QMIXWrapper(_BaseCoordinator):
    def __init__(self, n_agents: int, device):
        self._coord = QMIXCoordinator(n_agents=n_agents, device=device)

    def act(self, obs_list, epsilon=0.0):
        return self._coord.select_actions(obs_list, epsilon=epsilon)

    def push(self, gs, obs, acts, reward, ngs, nobs, done, weather=0):
        self._coord.push_transition(
            gs, np.stack(obs), np.array(acts), reward,
            ngs, np.stack(nobs), done, weather_state=weather
        )

    def train_step(self):
        return self._coord.train()

    def sync_targets(self):
        self._coord.sync_targets()


# =============================================================================
# EPISODE RUNNER (with the earlier next-obs fix retained)
# =============================================================================

def run_episode(env: QuickCommerceEnv, network, coordinator: _BaseCoordinator,
                epsilon: float, is_train: bool) -> dict:
    env.reset()
    route_counts = {0: 0, 1: 0, 2: 0}

    while env.t < config.EPISODE_LENGTH:
        idle    = env.idle_riders()
        pending = env.pending_orders()

        if not idle or not pending:
            env.step({})
            continue

        pending_sorted  = sorted(pending, key=lambda o: o.urgency(env.t), reverse=True)
        pairs           = list(zip(idle, pending_sorted))
        rider_order_map = {r.id: o for r, o in pairs}

        obs_list = []
        for rider in env.riders:
            order = rider_order_map.get(rider.id)
            if order is not None and rider.is_idle:
                obs_list.append(
                    build_agent_state(rider, order, env.traffic, env.weather, env.t, network)
                )
            else:
                obs_list.append(np.zeros(config.STATE_DIM, dtype=np.float32))

        gs_before      = build_global_state(env.riders, pending, env.traffic, env.weather, env.t)
        weather_now    = env.weather
        actions        = coordinator.act(obs_list, epsilon=epsilon)
        path_decisions = {r.id: actions[r.id] for r, o in pairs}

        for r, o in pairs:
            route_counts[actions[r.id]] = route_counts.get(actions[r.id], 0) + 1

        _, reward, done, _ = env.step(path_decisions)

        # ── next-obs fix (retained from previous round) ──────────────────────
        next_idle    = env.idle_riders()
        next_pending = env.pending_orders()
        if next_idle and next_pending:
            next_pending_sorted = sorted(next_pending, key=lambda o: o.urgency(env.t), reverse=True)
            next_pairs = list(zip(next_idle, next_pending_sorted))
            next_rider_order_map = {r.id: o for r, o in next_pairs}
        else:
            next_rider_order_map = {}

        nobs_list = []
        for rider in env.riders:
            order = next_rider_order_map.get(rider.id)
            if order is not None and rider.is_idle:
                nobs_list.append(
                    build_agent_state(rider, order, env.traffic, env.weather, env.t, network)
                )
            else:
                nobs_list.append(np.zeros(config.STATE_DIM, dtype=np.float32))
        nobs_arr = np.stack(nobs_list)

        gs_after = build_global_state(env.riders, next_pending, env.traffic, env.weather, env.t)

        if is_train:
            coordinator.push(
                gs_before, np.stack(obs_list), np.array(actions),
                reward, gs_after, nobs_arr, bool(done), weather=weather_now,
            )
            coordinator.train_step()

        if done:
            break

    delivered = [o for o in env.orders.values() if o.delivered]
    on_time   = sum(1 for o in delivered if not o.is_late())

    return {
        "J":        env.compute_J(),
        "on_time":  on_time / max(1, len(delivered)),
        "dist":     sum(r.total_distance for r in env.riders),
        "routes":   route_counts,
    }


# =============================================================================
# TRAINING / EVAL / MAIN  (unchanged from previous version)
# =============================================================================

METHOD_LABELS = {
    "DQN":   "DQN (single net, no agent-id)",
    "IQL":   "IQL (shared wts, no coord.)",
    "VDN":   "VDN (shared wts, additive)",
    "MAPPO": "MAPPO (shared actor+critic)",
    "QMIX":  "QMIX MARL (Ours)",
}

def build_coordinator(method: str, device) -> _BaseCoordinator:
    if method == "DQN":
        return DQNCoordinator(device=device)
    elif method == "IQL":
        return IQLCoordinator(n_agents=config.N_RIDERS, device=device)
    elif method == "VDN":
        return VDNCoordinator(n_agents=config.N_RIDERS, device=device)
    elif method == "MAPPO":
        return MAPPOCoordinator(n_agents=config.N_RIDERS, device=device)
    elif method == "QMIX":
        return QMIXWrapper(n_agents=config.N_RIDERS, device=device)
    else:
        raise ValueError(f"Unknown method: {method}")


def train_method(method: str, episodes: int, env, network, device, log_steps=100):
    coord = build_coordinator(method, device)
    eps   = config.EPSILON_START
    AJ=[]; AOT=[]
    label = METHOD_LABELS.get(method, method)

    log(f"Training {label} ({episodes} episodes) on {device}...")

    with tqdm(total=episodes, desc=f"  {label[:30]:<30} train",
              unit="ep", ncols=110, leave=True) as bar:
        for ep in range(1, episodes + 1):
            result = run_episode(env, network, coord, eps, is_train=True)
            AJ.append(result["J"]); AOT.append(result["on_time"])

            if method != "MAPPO":
                eps = max(config.EPSILON_END, eps * config.EPSILON_DECAY)

            if ep % config.TARGET_UPDATE_FREQ == 0:
                coord.sync_targets()

            w = min(log_steps, ep)
            bar.set_postfix(J=f"{np.mean(AJ[-w:]):.1f}",
                            OT=f"{np.mean(AOT[-w:])*100:.1f}%", eps=f"{eps:.4f}")
            bar.update(1)

            if ep % log_steps == 0 or ep == 1 or ep == episodes:
                log(f"[{label} | train ep{ep}/{episodes}] "
                    f"J({w}ep)={np.mean(AJ[-w:]):.2f}  "
                    f"OT={np.mean(AOT[-w:])*100:.1f}%  ε={eps:.4f}")

    return coord


def evaluate_method(method, coord, env, network, eval_ep, log_steps=25):
    label = METHOD_LABELS.get(method, method)
    Js=[]; OTs=[]; Ds=[]; RC={0:0,1:0,2:0}

    with tqdm(total=eval_ep, desc=f"  {label[:30]:<30}  eval",
              unit="ep", ncols=110, leave=False) as bar:
        for ev in range(1, eval_ep + 1):
            result = run_episode(env, network, coord, epsilon=0.0, is_train=False)
            Js.append(result["J"]); OTs.append(result["on_time"]); Ds.append(result["dist"])
            for k, v in result["routes"].items():
                RC[k] = RC.get(k, 0) + v
            bar.set_postfix(J=f"{result['J']:.1f}", OT=f"{result['on_time']*100:.1f}%")
            bar.update(1)
            if ev % log_steps == 0 or ev == eval_ep:
                log(f"[{label} | eval ep{ev}/{eval_ep}] "
                    f"J={np.mean(Js):.2f}  OT={np.mean(OTs)*100:.1f}%")

    tot = sum(RC.values()) + 1e-9
    return {
        "mean_J":  float(np.mean(Js)),
        "std_J":   float(np.std(Js)),
        "mean_OT": float(np.mean(OTs)) * 100,
        "mean_dist": float(np.mean(Ds)),
        "all_J":   Js,
        "route_pct": {k: v / tot * 100 for k, v in RC.items()},
    }


def evaluate_all(city, episodes=3000, eval_ep=100, methods=None, log_steps=100):
    _setup_logger(city)
    dev_name = (f"cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name()})"
                if torch.cuda.is_available() else "CPU")
    log(f"{'='*65}")
    log(f"RL BASELINE COMPARISON")
    log(f"City   : {city}  |  Train: {episodes} ep  |  Eval: {eval_ep} ep")
    log(f"Device : {dev_name}")
    log(f"NOTE: DQN/IQL/VDN/QMIX all use IDENTICAL 30% flood-prioritised "
        f"replay sampling. MAPPO is on-policy (no replay buffer confound).")
    log(f"{'='*65}")

    config.load_city(city)
    network = RoadNetwork(use_cache=True)
    env     = QuickCommerceEnv(network=network)

    all_methods = methods or ["DQN", "IQL", "VDN", "MAPPO", "QMIX"]
    results     = {}

    for method in all_methods:
        log(f"\n{'─'*65}")
        log(f"METHOD: {METHOD_LABELS.get(method, method)}")
        log(f"{'─'*65}")

        coord = train_method(method, episodes, env, network, device=DEVICE, log_steps=log_steps)

        log(f"Evaluating {method} ({eval_ep} episodes, ε=0)...")
        result = evaluate_method(method, coord, env, network, eval_ep, log_steps=25)
        results[method] = result

        log(f"RESULT [{method}]: J={result['mean_J']:.2f} ±{result['std_J']:.2f}  "
            f"OT={result['mean_OT']:.1f}%  "
            f"Routes R0={result['route_pct'].get(0,0):.0f}% "
            f"R1={result['route_pct'].get(1,0):.0f}% "
            f"R2={result['route_pct'].get(2,0):.0f}%")

    log(f"\n{'='*75}")
    log(f"FINAL COMPARISON — {city.upper()}")
    log(f"{'='*75}")
    qmix_J   = results.get("QMIX", {}).get("mean_J", None)
    base_key = all_methods[0]
    base_J   = results[base_key]["mean_J"]

    log(f"{'Method':<35} {'J (RS)':>9} {'+-':>7} {'OnTime%':>8} {'vs '+base_key:>12} {'vs QMIX':>9}")
    log("-" * 84)
    for m, r in results.items():
        vs_base = (base_J - r["mean_J"]) / base_J * 100
        vs_qmix = (qmix_J - r["mean_J"]) / qmix_J * 100 if qmix_J else float("nan")
        marker  = "  <-- OURS" if m == "QMIX" else ""
        log(f"  {METHOD_LABELS.get(m,m):<33} {r['mean_J']:>8.2f} "
            f"{r['std_J']:>7.2f} {r['mean_OT']:>7.1f}% "
            f"{vs_base:>+11.1f}% {vs_qmix:>+8.1f}%{marker}")
    log(f"{'='*75}")

    os.makedirs("results", exist_ok=True)
    out_path = f"results/rl_baselines_{city}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    log(f"\nSaved → {out_path}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RL Baseline Comparison")
    ap.add_argument("--city", default="bangalore",
                    choices=["bangalore","chennai","hyderabad","delhi","mumbai"])
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--eval_ep",  type=int, default=100)
    ap.add_argument("--methods",  nargs="+",
                    choices=["DQN","IQL","VDN","MAPPO","QMIX"],
                    default=["DQN","IQL","VDN","MAPPO","QMIX"])
    ap.add_argument("--log_steps", type=int, default=100)
    args = ap.parse_args()
    evaluate_all(
        city      = args.city,
        episodes  = args.episodes,
        eval_ep   = args.eval_ep,
        methods   = args.methods,
        log_steps = args.log_steps,
    )