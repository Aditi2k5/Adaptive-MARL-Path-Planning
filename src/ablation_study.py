"""
ablation_study.py — Complete corrected version (v3)

Fixes vs previous version:
  - Real next-observations built from actual post-step state (was
    unconditionally zero for every rider, matching the bug already
    fixed in train.py and run_rl_baselines.py)
  - IQL now uses SharedQNet (agent-id-conditioned, shared weights),
    matching the methodology used in run_rl_baselines.py, so ablation's
    "IQL (no mixing)" isolates coordination only, not sample efficiency
  - theta4/theta5 default to the CALIBRATED values (1.5 / 0.75), not the
    raw config values (8.0 / 6.0) which dominate the objective and
    produce inflated, scale-artifact deltas. Pass --theta4 8.0 --theta5
    6.0 explicitly if you want the uncalibrated config defaults instead.

Run:
    python ablation_study.py --city chennai --episodes 3000 --eval_ep 100
    python ablation_study.py --city delhi   --episodes 3000 --eval_ep 100
"""

from __future__ import annotations
import argparse, logging, os, pickle, random
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import config
from qmix import QMIXCoordinator
from agent import build_agent_state, build_global_state
from environment import QuickCommerceEnv
from road_network import RoadNetwork

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required. This script is configured to run on GPU only.")
DEVICE = torch.device("cuda")

# ── Recommended calibrated theta values (see ABLATION_BUG_ANALYSIS.md) ──────
CALIBRATED_THETA4 = 1.5
CALIBRATED_THETA5 = 0.75

LOGGER = logging.getLogger("ablation")

def _setup_logger(city: str) -> None:
    os.makedirs("results", exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    for h in list(LOGGER.handlers):
        LOGGER.removeHandler(h); h.close()
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(f"results/ablation_{city}.log", "w", "utf-8"); fh.setFormatter(fmt)
    LOGGER.addHandler(sh); LOGGER.addHandler(fh)

def log(msg: str) -> None:
    LOGGER.info(msg)

def _device_banner() -> str:
    idx = torch.cuda.current_device()
    return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"


ABLATIONS = {
    "Full QMIX":            {"theta4": True,  "theta5": True,  "flood_replay": True,  "use_mixer": True},
    "No θ4 (no weather)":   {"theta4": False, "theta5": True,  "flood_replay": True,  "use_mixer": True},
    "No θ5 (no breakdown)": {"theta4": True,  "theta5": False, "flood_replay": True,  "use_mixer": True},
    "No flood replay":      {"theta4": True,  "theta5": True,  "flood_replay": False, "use_mixer": True},
    "IQL (no mixing)":      {"theta4": True,  "theta5": True,  "flood_replay": True,  "use_mixer": False},
}


# =============================================================================
# Shared, agent-ID-conditioned Q-network (matches run_rl_baselines.py)
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


def _agent_id_tensor(n: int, device) -> torch.Tensor:
    return torch.arange(n, dtype=torch.long, device=device)


# =============================================================================
# Reward / J math — per-delivery, weather-snapshot-correct
# =============================================================================

def compute_step_reward(dispatched: List[Tuple], weather_at_dispatch: int,
                        theta4_on: bool, theta5_on: bool,
                        theta4_val: float, theta5_val: float,
                        riders_health: dict) -> float:
    reward = 0.0
    for order, route_idx, rider_id in dispatched:
        route = order.routes[route_idx]
        dist  = route.distance_km * 2

        travel    = config.C_KM * dist
        breakdown = (theta5_val * (1.0 - riders_health[rider_id]) * dist
                     if theta5_on else 0.0)
        weather_r = (route.weather_risk_penalty(weather_at_dispatch) * (theta4_val / config.THETA_4)
                     if theta4_on else 0.0)
        late      = (config.P_LATE * order.delay() if order.is_late() else 0.0)

        reward -= (travel + breakdown + weather_r + late)
    return reward / config.REWARD_SCALE


def compute_episode_J(env, delivered_log: List[Tuple],
                      theta4_on: bool, theta5_on: bool,
                      theta4_val: float, theta5_val: float) -> float:
    J = 0.0
    for order, route_idx, w_snap, h_j in delivered_log:
        route = order.routes[route_idx]
        dist  = route.distance_km * 2

        J += config.C_KM * dist
        if theta5_on:
            J += theta5_val * (1.0 - h_j) * dist
        if theta4_on:
            J += route.weather_risk_penalty(w_snap) * (theta4_val / config.THETA_4)
        if order.is_late():
            J += config.P_LATE * order.delay()

    for o in env.pending_orders():
        J += config.THETA_1 * o.urgency(env.t)

    return max(0.0, J)


# =============================================================================
# FIX: build real next-observations (same logic as train.py / run_rl_baselines.py)
# =============================================================================

def build_next_rider_order_map(env: QuickCommerceEnv) -> dict:
    next_idle    = env.idle_riders()
    next_pending = env.pending_orders()
    if not next_idle or not next_pending:
        return {}
    next_pending_sorted = sorted(next_pending, key=lambda o: o.urgency(env.t), reverse=True)
    next_pairs = list(zip(next_idle, next_pending_sorted))
    return {r.id: o for r, o in next_pairs}


def build_next_obs(env: QuickCommerceEnv, network) -> np.ndarray:
    """Real next-observations, not unconditional zeros."""
    next_rider_order_map = build_next_rider_order_map(env)
    nobs_list = []
    for rider in env.riders:
        order = next_rider_order_map.get(rider.id)
        if order is not None and rider.is_idle:
            nobs_list.append(
                build_agent_state(rider, order, env.traffic, env.weather, env.t, network)
            )
        else:
            nobs_list.append(np.zeros(config.STATE_DIM, dtype=np.float32))
    return np.stack(nobs_list)


# =============================================================================
# FIX: IQL coordinator using shared, agent-ID-conditioned weights
# (matches run_rl_baselines.py methodology — ablates ONLY coordination,
# not sample efficiency)
# =============================================================================

class IQLCoordinator:
    def __init__(self, n_agents: int, device):
        self.n      = n_agents
        self.device = device
        self.q_net  = SharedQNet(n_agents=n_agents).to(device)
        self.t_net  = SharedQNet(n_agents=n_agents).to(device)
        self.t_net.load_state_dict(self.q_net.state_dict())
        self.opt    = optim.Adam(self.q_net.parameters(), lr=config.LEARNING_RATE)
        self.buf: deque       = deque(maxlen=config.REPLAY_BUFFER_SIZE)
        self.flood_buf: deque = deque(maxlen=5000)
        self._agent_ids = _agent_id_tensor(n_agents, device)

    def select_actions(self, obs_list, epsilon=0.0):
        obs_t = torch.FloatTensor(np.stack(obs_list)).to(self.device)
        with torch.no_grad():
            greedy = self.q_net(obs_t, self._agent_ids).argmax(dim=1).cpu().numpy()
        actions = []
        for i in range(self.n):
            if random.random() < epsilon:
                actions.append(random.randint(0, config.ACTION_DIM - 1))
            else:
                actions.append(int(greedy[i]))
        return actions

    def push_transition(self, gs, obs, actions, reward, ngs, nobs, done, weather_state=0):
        for i in range(self.n):
            item = (obs[i], i, actions[i], reward, nobs[i], float(done))
            self.buf.append(item)
            if weather_state >= 2:
                self.flood_buf.append(item)

    def _sample(self, batch_size: int):
        if len(self.buf) < batch_size:
            return None
        n_flood = min(int(batch_size * 0.30), len(self.flood_buf))
        n_norm  = batch_size - n_flood
        batch = (random.sample(list(self.flood_buf), n_flood) if n_flood > 0 else []) + \
                random.sample(list(self.buf), n_norm)
        random.shuffle(batch)
        return batch

    def train(self):
        batch = self._sample(config.BATCH_SIZE)
        if batch is None:
            return None
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


def build_coordinator(use_mixer: bool, device):
    if use_mixer:
        return QMIXCoordinator(n_agents=config.N_RIDERS, device=device)
    return IQLCoordinator(n_agents=config.N_RIDERS, device=device)


# =============================================================================
# Episode runners
# =============================================================================

def run_train_episode(env, network, coord, flags,
                      theta4_val, theta5_val, eps) -> dict:
    env.reset()
    delivered_log: List[Tuple] = []

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
                obs_list.append(build_agent_state(rider, order, env.traffic, env.weather, env.t, network))
            else:
                obs_list.append(np.zeros(config.STATE_DIM, dtype=np.float32))

        weather_before = env.weather
        gs_before = build_global_state(env.riders, pending, env.traffic, env.weather, env.t)

        actions = coord.select_actions(obs_list, epsilon=eps)
        path_decisions = {r.id: actions[r.id] for r, o in pairs}

        dispatched = []
        for rider, order in pairs:
            a = actions[rider.id]
            dispatched.append((order, a, rider.id))
            delivered_log.append((order, a, weather_before, rider.health))

        _, _, done, _ = env.step(path_decisions)

        # ── FIX: real next-observations, not unconditional zeros ─────────────
        nobs_arr = build_next_obs(env, network)
        gs_after = build_global_state(env.riders, env.pending_orders(),
                                      env.traffic, env.weather, env.t)

        riders_health = {r.id: r.health for r in env.riders}
        step_r = compute_step_reward(
            dispatched, weather_before,
            flags["theta4"], flags["theta5"],
            theta4_val, theta5_val,
            riders_health,
        )

        coord.push_transition(
            gs_before, np.stack(obs_list), np.array(actions),
            step_r, gs_after, nobs_arr, bool(done),
            weather_state=weather_before if flags["flood_replay"] else 0,
        )
        coord.train()

        if done:
            break

    J  = compute_episode_J(env, delivered_log, flags["theta4"], flags["theta5"], theta4_val, theta5_val)
    d  = [o for o in env.orders.values() if o.delivered]
    ot = sum(1 for o in d if not o.is_late()) / max(1, len(d))
    return {"J": J, "on_time": ot}


def run_eval_episode(env, network, coord, flags, theta4_val, theta5_val) -> dict:
    env.reset()
    delivered_log: List[Tuple] = []

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
                obs_list.append(build_agent_state(rider, order, env.traffic, env.weather, env.t, network))
            else:
                obs_list.append(np.zeros(config.STATE_DIM, dtype=np.float32))

        weather_before = env.weather
        actions = coord.select_actions(obs_list, epsilon=0.0)
        path_decisions = {r.id: actions[r.id] for r, o in pairs}

        for rider, order in pairs:
            delivered_log.append((order, actions[rider.id], weather_before, rider.health))

        _, _, done, _ = env.step(path_decisions)
        if done:
            break

    J  = compute_episode_J(env, delivered_log, flags["theta4"], flags["theta5"], theta4_val, theta5_val)
    d  = [o for o in env.orders.values() if o.delivered]
    ot = sum(1 for o in d if not o.is_late()) / max(1, len(d))
    return {"J": J, "on_time": ot}


# =============================================================================
# Main ablation runner
# =============================================================================

def run_ablation(city: str, episodes: int = 3000, eval_ep: int = 100,
                 theta4_val: Optional[float] = None,
                 theta5_val: Optional[float] = None,
                 log_every: int = 100) -> dict:

    _setup_logger(city)
    config.load_city(city)

    theta4_val = theta4_val if theta4_val is not None else CALIBRATED_THETA4
    theta5_val = theta5_val if theta5_val is not None else CALIBRATED_THETA5

    log(f"Device : {_device_banner()}")
    log(f"City   : {city}")
    log(f"Train  : {episodes} episodes  |  Eval: {eval_ep} episodes")
    log(f"θ4={theta4_val}  θ5={theta5_val}  "
        f"(config raw defaults: θ4={config.THETA_4}, θ5={config.THETA_5})")
    if theta4_val == config.THETA_4 or theta5_val == config.THETA_5:
        log("  *** WARNING: using UNCALIBRATED theta value(s) — expect inflated,  ***")
        log("  *** scale-artifact deltas for the theta4/theta5 ablation rows.     ***")
    log("  NOTE: IQL now uses SharedQNet (agent-id-conditioned, shared weights),")
    log("        matching run_rl_baselines.py methodology.")
    log("  NOTE: next-observations are built from real post-step state, not zeros.")

    network = RoadNetwork(use_cache=True)
    env     = QuickCommerceEnv(network=network)

    results = {}

    for abl_name, flags in ABLATIONS.items():
        log(f"\n{'='*65}")
        log(f"Starting: {abl_name}")
        log(f"  θ4={flags['theta4']}  θ5={flags['theta5']}  "
            f"flood_replay={flags['flood_replay']}  mixer={flags['use_mixer']}")
        log(f"{'='*65}")

        coord = build_coordinator(flags["use_mixer"], DEVICE)
        eps = config.EPSILON_START
        AJ: List[float] = []; AOT: List[float] = []
        window = 100

        with tqdm(total=episodes, desc=f"  {abl_name[:28]:<28} train",
                  unit="ep", ncols=110, leave=True) as bar:
            for ep in range(1, episodes + 1):
                r = run_train_episode(env, network, coord, flags, theta4_val, theta5_val, eps)
                AJ.append(r["J"]); AOT.append(r["on_time"])

                eps = max(config.EPSILON_END, eps * config.EPSILON_DECAY)
                if ep % config.TARGET_UPDATE_FREQ == 0:
                    coord.sync_targets()

                w_J  = np.mean(AJ[-window:])
                w_OT = np.mean(AOT[-window:]) * 100
                bar.set_postfix(J=f"{w_J:.1f}", OT=f"{w_OT:.1f}%", eps=f"{eps:.4f}")
                bar.update(1)

                if ep % log_every == 0 or ep == 1 or ep == episodes:
                    log(f"  [train {ep:5d}/{episodes}] "
                        f"J({window}ep)={w_J:.2f}  OT={w_OT:.1f}%  ε={eps:.4f}")

        log(f"  Evaluating {abl_name} ({eval_ep} episodes, ε=0)...")
        Js: List[float] = []; OTs: List[float] = []
        with tqdm(total=eval_ep, desc=f"  {abl_name[:28]:<28} eval",
                  unit="ep", ncols=110, leave=False) as bar:
            for ev in range(1, eval_ep + 1):
                r = run_eval_episode(env, network, coord, flags, theta4_val, theta5_val)
                Js.append(r["J"]); OTs.append(r["on_time"])
                bar.set_postfix(J=f"{r['J']:.1f}", OT=f"{r['on_time']*100:.1f}%")
                bar.update(1)
                if ev % 25 == 0 or ev == eval_ep:
                    log(f"  [eval {ev:3d}/{eval_ep}] "
                        f"J={np.mean(Js):.2f}  OT={np.mean(OTs)*100:.1f}%")

        results[abl_name] = {
            "mean_J":  float(np.mean(Js)),
            "std_J":   float(np.std(Js)),
            "mean_OT": float(np.mean(OTs)) * 100,
            "all_J":   Js,
            "train_J": AJ,
        }
        log(f"  RESULT: J={np.mean(Js):.2f} ± {np.std(Js):.2f}  OT={np.mean(OTs)*100:.1f}%")

    log(f"\n{'='*65}")
    log(f"ABLATION STUDY — {city.upper()}  (θ4={theta4_val}, θ5={theta5_val})")
    log(f"{'='*65}")
    base = results["Full QMIX"]["mean_J"]
    log(f"{'Configuration':<30} {'J (₹)':>9} {'Std':>7} {'OnTime':>8} {'Δ vs Full':>10}")
    log("-" * 65)
    for name, r in results.items():
        delta  = (r["mean_J"] - base) / base * 100
        marker = " ← FULL" if name == "Full QMIX" else ""
        log(f"  {name:<28} {r['mean_J']:>8.2f} {r['std_J']:>7.2f} "
            f"{r['mean_OT']:>7.1f}% {delta:>+9.1f}%{marker}")
    log(f"{'='*65}")
    log("  Positive Δ = removing that component made J WORSE (component was helping)")
    log("  Negative Δ = removing that component made J BETTER (component was hurting)")

    os.makedirs("results", exist_ok=True)
    out = f"results/ablation_{city}.pkl"
    with open(out, "wb") as f:
        pickle.dump(results, f)
    log(f"\n  Saved → {out}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ablation study for QMIX adaptive path planning")
    ap.add_argument("--city",      default="bangalore",
                    choices=["bangalore", "chennai", "hyderabad", "delhi", "mumbai"])
    ap.add_argument("--episodes",  type=int, default=3000)
    ap.add_argument("--eval_ep",   type=int, default=100)
    ap.add_argument("--theta4",    type=float, default=None,
                    help=f"Override THETA_4. Default: calibrated value {CALIBRATED_THETA4} "
                         f"(config raw default {config.THETA_4} is uncalibrated).")
    ap.add_argument("--theta5",    type=float, default=None,
                    help=f"Override THETA_5. Default: calibrated value {CALIBRATED_THETA5} "
                         f"(config raw default {config.THETA_5} is uncalibrated).")
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()

    run_ablation(
        city       = args.city,
        episodes   = args.episodes,
        eval_ep    = args.eval_ep,
        theta4_val = args.theta4,
        theta5_val = args.theta5,
        log_every  = args.log_every,
    )