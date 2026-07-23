from __future__ import annotations
import os
import random
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import config
from data_structures import Order, Rider


def build_agent_state(rider: Rider,
                      order: Order,
                      traffic: float,
                      weather: int,
                      current_time: float,
                      network) -> np.ndarray:
    feats: List[float] = []

    feats.append(rider.health)                           # h_j
    feats.append(traffic / 1.3)                          # τ_t normalised
    feats.append(weather / 2.0)                          # w_t normalised


    for route in order.routes:
        tm  = route.travel_time_minutes(traffic, weather) / 30.0   # normalised
        dm  = route.distance_km / config.SERVICE_RADIUS_KM         # normalised
        fr  = route.flood_risk                                       # [0, 1]
        wrp = route.weather_risk_penalty(weather) / 10.0            # normalised
        feats.extend([tm, dm, fr, wrp])

    # Pad route features if fewer than K_ROUTES routes present
    for _ in range(config.K_ROUTES - len(order.routes)):
        feats.extend([0.0, 0.0, 0.0, 0.0])

    u   = order.urgency(current_time)                               # u_i(t)
    ttd = max(0.0, order.deadline_time - current_time) / 30.0      # normalised
    dist_to_customer = network.road_distance(
        config.DARK_STORE_LOCATION, order.location
    ) / config.SERVICE_RADIUS_KM

    feats.append(u)
    feats.append(ttd)
    feats.append(dist_to_customer)

    while len(feats) < config.STATE_DIM:
        feats.append(0.0)

    return np.array(feats[:config.STATE_DIM], dtype=np.float32)


def build_global_state(riders: List[Rider],
                       pending_orders: List[Order],
                       traffic: float,
                       weather: int,
                       current_time: float) -> np.ndarray:
    g: List[float] = []

    for r in riders:
        g.append(r.health)
        g.append(1.0 if r.is_idle else 0.0)
        eta = max(0.0, r.return_at - current_time) / 30.0
        g.append(eta)

    g.append(traffic / 1.3)
    g.append(weather / 2.0)
    g.append(current_time / config.EPISODE_LENGTH)

    urgencies = sorted(
        [o.urgency(current_time) for o in pending_orders], reverse=True
    )[:3]
    while len(urgencies) < 3:
        urgencies.append(0.0)
    g.extend(urgencies)

    g.append(min(len(pending_orders) / 8.0, 1.0))
    g.append(sum(1 for r in riders if r.is_idle) / max(1, len(riders)))

    while len(g) < config.GLOBAL_STATE_DIM:
        g.append(0.0)

    return np.array(g[:config.GLOBAL_STATE_DIM], dtype=np.float32)

class QNet(nn.Module):
    def __init__(self,
                 state_dim: int = config.STATE_DIM,
                 action_dim: int = config.ACTION_DIM,
                 hidden:     int = config.HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, 64),       nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class RiderAgent:


    def __init__(self, rider_id: int, device: torch.device = None):
        self.rider_id = rider_id
        self.epsilon  = config.EPSILON_START
        if device is None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU is required. This project is configured to run on CUDA only.")
            device = torch.device("cuda")
        self.device   = device

        self.q_net  = QNet().to(self.device)
        self.t_net  = QNet().to(self.device)   # target network (frozen, updated periodically)
        self.t_net.load_state_dict(self.q_net.state_dict())

        self.optimiser = optim.Adam(
            self.q_net.parameters(), lr=config.LEARNING_RATE
        )


    def act(self, obs: np.ndarray,
            epsilon: Optional[float] = None) -> int:
        eps = epsilon if epsilon is not None else self.epsilon

        if random.random() < eps:
            return random.randint(0, config.ACTION_DIM - 1)

        with torch.no_grad():
            q = self.q_net(torch.FloatTensor(obs).unsqueeze(0).to(self.device))[0]
            return int(q.argmax().item())

    def get_q_values(self, obs: np.ndarray) -> torch.Tensor:
        """Return full Q-value tensor; used by QMIX mixer during training."""
        with torch.no_grad():
            return self.q_net(
                torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            ).squeeze(0)


    def learn(self,
              state:      np.ndarray,
              action:     int,
              reward:     float,
              next_state: np.ndarray,
              done:       bool) -> float:

        s   = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        s_  = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        a   = torch.LongTensor([[action]]).to(self.device)
        r   = torch.FloatTensor([reward]).to(self.device)
        d   = torch.FloatTensor([float(done)]).to(self.device)

        q_curr = self.q_net(s).gather(1, a).squeeze()
        with torch.no_grad():
            q_next = self.t_net(s_).max(1)[0]
            q_tgt  = r + config.GAMMA * q_next * (1.0 - d)

        loss = nn.MSELoss()(q_curr, q_tgt)
        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimiser.step()
        return loss.item()


    @staticmethod
    def compute_reward(route_distance: float,
                       late_delay_min:  float,
                       health:          float,
                       weather_risk:    float) -> float:
        cost  = config.C_KM   * route_distance * 2        # out + return
        cost += config.P_LATE * late_delay_min
        cost += config.THETA_5 * (1.0 - health) * route_distance * 2
        cost += weather_risk
        return -cost


    def sync_target(self) -> None:
        self.t_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self) -> None:
        self.epsilon = max(config.EPSILON_END,
                           self.epsilon * config.EPSILON_DECAY)


    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "q":       self.q_net.state_dict(),
            "t":       self.t_net.state_dict(),
            "opt":     self.optimiser.state_dict(),
            "epsilon": self.epsilon,
        }, path)

    def load(self, path: str, device: torch.device = None) -> None:
        if device is not None:
            self.device = device
        ck = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ck["q"])
        self.t_net.load_state_dict(ck["t"])
        self.q_net = self.q_net.to(self.device)
        self.t_net = self.t_net.to(self.device)
        self.optimiser.load_state_dict(ck["opt"])
        self.epsilon = ck["epsilon"]


if __name__ == "__main__":
    import random as _r; _r.seed(42)
    import numpy as _np; _np.random.seed(42)

    from data_structures import sample_order, make_fleet
    from road_network import RoadNetwork

    print("agent.py — self-test\n")

    net   = RoadNetwork(use_cache=True)
    fleet = make_fleet(1)
    order = sample_order(0, 0.0, net)
    rider = fleet[0]

    print(f"Order has {len(order.routes)} route alternatives:")
    for r in order.routes:
        t_clear = r.travel_time_minutes(1.0, 0)
        t_rain  = r.travel_time_minutes(1.3, 2)
        print(f"  Route {r.id} ({r.route_type}): "
              f"dist={r.distance_km:.2f}km  "
              f"clear={t_clear:.1f}min  peak+rain={t_rain:.1f}min  "
              f"flood={r.flood_risk:.2f}")

    # Build state in clear weather
    obs_clear = build_agent_state(rider, order, 1.0, 0, 5.0, net)
    print(f"\nState (clear weather): shape={obs_clear.shape}")
    print(f"  health={obs_clear[0]:.2f}  traffic={obs_clear[1]:.2f}  weather={obs_clear[2]:.2f}")

    # Build state in heavy rain
    obs_rain = build_agent_state(rider, order, 1.3, 2, 5.0, net)
    print(f"State (heavy rain):  shape={obs_rain.shape}")
    print(f"  Route 0 travel time feature: {obs_rain[3]:.3f}")
    print(f"  Route 1 travel time feature: {obs_rain[7]:.3f}")
    print(f"  Route 2 travel time feature: {obs_rain[11]:.3f}  ← should be HIGH in rain")

    agent = RiderAgent(rider_id=0)

    # Action in clear weather (should prefer shortcut or arterial)
    a_clear = agent.act(obs_clear, epsilon=0.0)
    # Action in rain (random at first, but trained agent should avoid route 2)
    a_rain  = agent.act(obs_rain, epsilon=0.0)
    print(f"\nGreedy action (clear): Route {a_clear} ({order.routes[a_clear].route_type})")
    print(f"Greedy action (rain):  Route {a_rain} ({order.routes[a_rain].route_type})")
    print("(Before training, actions are essentially random — training changes this)")

    # One learning step
    reward = RiderAgent.compute_reward(
        route_distance=order.routes[a_clear].distance_km,
        late_delay_min=0.0,
        health=rider.health,
        weather_risk=order.routes[a_clear].weather_risk_penalty(0)
    )
    loss = agent.learn(obs_clear, a_clear, reward, obs_rain, done=False)
    print(f"\nTraining loss: {loss:.5f}")

    # Global state
    gs = build_global_state(fleet, [order], 1.0, 0, 5.0)
    print(f"Global state: shape={gs.shape}")

    print("\n✓ agent.py OK")