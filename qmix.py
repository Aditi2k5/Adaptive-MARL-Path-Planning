from __future__ import annotations
import os
import random
from collections import deque
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import config
from agent import QNet, RiderAgent


class MixingNetwork(nn.Module):

    def __init__(self,
                 n_agents:  int = config.N_RIDERS,
                 state_dim: int = config.GLOBAL_STATE_DIM,
                 embed:     int = config.QMIX_EMBED_DIM):
        super().__init__()
        self.n     = n_agents
        self.embed = embed

        # Hypernetworks for layer-1 weights and biases
        self.hw1 = nn.Sequential(
            nn.Linear(state_dim, embed), nn.ReLU(),
            nn.Linear(embed, n_agents * embed)
        )
        self.hb1 = nn.Linear(state_dim, embed)

        # Hypernetworks for layer-2 weights and biases
        self.hw2 = nn.Sequential(
            nn.Linear(state_dim, embed), nn.ReLU(),
            nn.Linear(embed, embed)
        )
        self.hb2 = nn.Sequential(
            nn.Linear(state_dim, embed), nn.ReLU(),
            nn.Linear(embed, 1)
        )

    def forward(self,
                agent_qs: torch.Tensor,   # (B, N)
                state:    torch.Tensor    # (B, global_state_dim)
                ) -> torch.Tensor:         # (B,)
        B = agent_qs.size(0)

        # Layer 1  — abs() enforces monotonicity
        w1 = torch.abs(self.hw1(state)).view(B, self.n, self.embed)
        b1 = self.hb1(state).view(B, 1, self.embed)
        h  = torch.relu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)  # (B,1,embed)

        # Layer 2  — abs() enforces monotonicity
        w2 = torch.abs(self.hw2(state)).view(B, self.embed, 1)
        b2 = self.hb2(state).view(B, 1, 1)
        q_tot = (torch.bmm(h, w2) + b2).squeeze(-1).squeeze(-1)     # (B,)

        return q_tot
    
class ReplayBuffer:
    def __init__(self, capacity: int = config.REPLAY_BUFFER_SIZE):
        self.buf = deque(maxlen=capacity)

    def push(self, transition: tuple) -> None:
        self.buf.append(transition)

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buf, batch_size)
        gs, obs, acts, rews, ngs, nobs, dones = zip(*batch)
        return (
            np.stack(gs),
            np.stack(obs),
            np.stack(acts),
            np.array(rews,  dtype=np.float32),
            np.stack(ngs),
            np.stack(nobs),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buf)


class QMIXCoordinator:
    """
    Manages N_RIDERS agents + mixing network + replay buffer.

    Public methods:
        select_actions(obs_list, epsilon)  →  List[int]   decentralised
        push_transition(...)               →  None
        train()                            →  Optional[float]  loss
        sync_targets()                     →  None
        decay_epsilon()                    →  None
        save(directory)  /  load(directory)
    """

    def __init__(self, n_agents: int = config.N_RIDERS, device: torch.device = None):
        self.n = n_agents
        self.device = device if device is not None else torch.device("cpu")

        # Per-agent Q-networks (one per rider)
        self.agents  = [RiderAgent(i, device=self.device) for i in range(n_agents)]

        # Online mixing network
        self.mixer   = MixingNetwork(n_agents).to(self.device)
        # Target mixing network (updated periodically)
        self.t_mixer = MixingNetwork(n_agents).to(self.device)
        self.t_mixer.load_state_dict(self.mixer.state_dict())

        # Target Q-networks (one per agent, updated periodically)
        self.t_qnets = [QNet().to(self.device) for _ in range(n_agents)]
        for i in range(n_agents):
            self.t_qnets[i].load_state_dict(self.agents[i].q_net.state_dict())

        # Single joint optimiser (trains all Q-nets + mixer together)
        params = list(self.mixer.parameters())
        for a in self.agents:
            params += list(a.q_net.parameters())
        self.opt = optim.Adam(params, lr=config.LEARNING_RATE)

        # Shared replay buffer
        self.buffer = ReplayBuffer()


    def select_actions(self,
                       obs_list: List[np.ndarray],
                       epsilon:  Optional[float] = None) -> List[int]:

        return [
            self.agents[i].act(obs_list[i], epsilon=epsilon)
            for i in range(self.n)
        ]

    def push_transition(self,
                        global_state:  np.ndarray,
                        obs:           np.ndarray,   # (N, STATE_DIM)
                        actions:       np.ndarray,   # (N,)
                        team_reward:   float,
                        next_global:   np.ndarray,
                        next_obs:      np.ndarray,   # (N, STATE_DIM)
                        done:          bool) -> None:
        self.buffer.push((
            global_state, obs, actions, float(team_reward),
            next_global, next_obs, float(done)
        ))


    def train(self) -> Optional[float]:
        if len(self.buffer) < config.BATCH_SIZE:
            return None

        gs, obs, acts, rews, ngs, nobs, dones = self.buffer.sample(
            config.BATCH_SIZE
        )

        gs_t   = torch.FloatTensor(gs).to(self.device)
        obs_t  = torch.FloatTensor(obs).to(self.device)     # (B, N, STATE_DIM)
        acts_t = torch.LongTensor(acts).to(self.device)     # (B, N)
        rews_t = torch.FloatTensor(rews).to(self.device)    # (B,)
        ngs_t  = torch.FloatTensor(ngs).to(self.device)
        nobs_t = torch.FloatTensor(nobs).to(self.device)
        done_t = torch.FloatTensor(dones).to(self.device)

        cur_qs = []
        for i, agent in enumerate(self.agents):
            q_i  = agent.q_net(obs_t[:, i, :])              # (B, ACTION_DIM)
            q_a  = q_i.gather(1, acts_t[:, i].unsqueeze(1)) # (B, 1)
            cur_qs.append(q_a)
        cur_qs_t = torch.cat(cur_qs, dim=1)                  # (B, N)
        q_tot    = self.mixer(cur_qs_t, gs_t)                # (B,)

        with torch.no_grad():
            tgt_qs = []
            for i in range(self.n):
                tq_i = self.t_qnets[i](nobs_t[:, i, :]).max(1)[0].unsqueeze(1)
                tgt_qs.append(tq_i)
            tgt_qs_t  = torch.cat(tgt_qs, dim=1)             # (B, N)
            q_tot_tgt = self.t_mixer(tgt_qs_t, ngs_t)        # (B,)
            targets   = rews_t + config.GAMMA * q_tot_tgt * (1.0 - done_t)

        loss = nn.MSELoss()(q_tot, targets)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for g in self.opt.param_groups for p in g["params"]], 10.0
        )
        self.opt.step()
        return loss.item()

    def sync_targets(self) -> None:
        """Hard-copy online weights → target networks."""
        self.t_mixer.load_state_dict(self.mixer.state_dict())
        for i in range(self.n):
            self.t_qnets[i].load_state_dict(self.agents[i].q_net.state_dict())

    def decay_epsilon(self) -> None:
        for agent in self.agents:
            agent.decay_epsilon()


    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        for i, agent in enumerate(self.agents):
            agent.save(os.path.join(directory, f"agent_{i}.pt"))
        torch.save(
            self.mixer.state_dict(),
            os.path.join(directory, "mixer.pt")
        )
        print(f"✓ Checkpoint saved → {directory}")

    def load(self, directory: str, device: torch.device = None) -> None:
        if device is not None:
            self.device = device
        for i, agent in enumerate(self.agents):
            agent.load(os.path.join(directory, f"agent_{i}.pt"), device=self.device)
        self.mixer.load_state_dict(
            torch.load(os.path.join(directory, "mixer.pt"), map_location=self.device)
        )
        # Move mixer to device
        self.mixer = self.mixer.to(self.device)
        self.t_mixer = self.t_mixer.to(self.device)
        self.sync_targets()
        print(f"✓ Checkpoint loaded ← {directory}")

if __name__ == "__main__":
    import random as _r; _r.seed(42)
    import numpy as _np; _np.random.seed(42)
    import torch as _t; _t.manual_seed(42)

    from agent import build_agent_state, build_global_state
    from data_structures import sample_order, make_fleet
    from road_network import RoadNetwork

    print("qmix.py — self-test\n")

    net    = RoadNetwork(use_cache=True)
    fleet  = make_fleet(config.N_RIDERS)
    orders = [sample_order(i, 0.0, net) for i in range(config.N_RIDERS)]

    coord = QMIXCoordinator()
    print(f"✓ Coordinator: {config.N_RIDERS} agents + mixing network")
    print(f"  State dim   : {config.STATE_DIM}")
    print(f"  Action dim  : {config.ACTION_DIM}  (3 route choices)")
    print(f"  Global dim  : {config.GLOBAL_STATE_DIM}")

    # Build per-agent observations
    obs_list = [
        build_agent_state(fleet[i], orders[i], 1.2, 1, 5.0, net)
        for i in range(config.N_RIDERS)
    ]

    # Decentralised action selection (random at first, ε=1.0)
    actions = coord.select_actions(obs_list, epsilon=1.0)
    print(f"\nRandom actions (before training): {actions}")
    for i, a in enumerate(actions):
        print(f"  Rider {i}: Route {a} ({orders[i].routes[a].route_type})")

    # Check for route conflicts
    conflicts = len(actions) - len(set(actions))
    print(f"\nRoute conflicts (before training): {conflicts}")
    print("  QMIX will learn to reduce conflicts via coordination")

    # Mixing network forward pass
    gs  = build_global_state(fleet, orders, 1.2, 1, 5.0)
    qs  = torch.rand(1, config.N_RIDERS)
    gs_t = torch.FloatTensor(gs).unsqueeze(0)
    q_tot = coord.mixer(qs, gs_t)
    print(f"\nQ_tot (scalar team value): {q_tot.item():.4f}")

    # Fill buffer and train
    obs_arr = _np.stack(obs_list)
    act_arr = _np.array(actions)
    for _ in range(config.BATCH_SIZE + 10):
        coord.push_transition(gs, obs_arr, act_arr, -5.0, gs, obs_arr, False)

    loss = coord.train()
    print(f"Training loss: {loss:.5f}")

    coord.decay_epsilon()
    print(f"Epsilon after decay: {coord.agents[0].epsilon:.4f}")

    print("\n✓ qmix.py OK")