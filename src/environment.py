from __future__ import annotations
import random
from typing import Dict, List, Optional, Tuple
import numpy as np
from src import config
from src.data_structures import Order, Rider, Route, make_fleet, sample_order
from src.road_network import RoadNetwork


class QuickCommerceEnv:
    REWARD_SCALE = 50.0

    def __init__(self, network: Optional[RoadNetwork] = None):
        print('Initialising QuickCommerceEnv (adaptive path planning) …')
        self.net     = network or RoadNetwork(use_cache=True)
        self.riders: List[Rider]      = []
        self.orders: Dict[int, Order] = {}
        self._nid    = 0
        self.t       = 0.0
        self.traffic = 1.0
        self.weather = 0
        self.completed = 0
        print('✓ Environment ready.')


    def reset(self) -> Dict:
        """Fresh episode: new riders, new orders, t=0."""
        self.t = 0.0; self._nid = 0
        self.orders = {}; self.completed = 0
        self.riders = make_fleet(config.N_RIDERS)
        for _ in range(max(1, np.random.poisson(3))):
            self._spawn()
        self._tick()
        return self.state()


    def step(self,
             path_decisions: Dict[int, int]   # {rider_id: route_index (0/1/2)}
             ) -> Tuple[Dict, float, bool, Dict]:

        for r in self.riders:
            if r.status in ('en_route', 'returning') and self.t >= r.return_at:
                r.status = 'idle'
                r.location = config.DARK_STORE_LOCATION
                r.current_order = None

        pending  = self.pending_orders()
        idle     = self.idle_riders()

        immediate_route_reward = 0.0
        sorted_pending = sorted(pending, key=lambda o: o.urgency(self.t), reverse=True)
        for rider, order in zip(idle, sorted_pending):
            route_idx = path_decisions.get(rider.id, 0)   # default: route 0
            rr = self._dispatch(rider, order, route_idx)
            immediate_route_reward += rr

        self.t += config.TIMESTEP_DURATION
        for _ in range(np.random.poisson(config.LAMBDA_ARRIVAL * config.TIMESTEP_DURATION / 60.0)):
            self._spawn()
        self._tick()

        done   = self.t >= config.EPISODE_LENGTH
        J      = self.compute_J()
        reward = (-J / self.REWARD_SCALE) + (immediate_route_reward / 10.0)

        return self.state(), reward, done, {
            'cost_J':       J,
            'completed':    self.completed,
            'on_time_rate': self._on_time_rate(),
            'n_pending':    len(self.pending_orders()),
        }


    def _dispatch(self, rider: Rider, order: Order, route_idx: int) -> float:

        route_idx = min(route_idx, len(order.routes) - 1)
        route     = order.routes[route_idx]

        # Travel time using THIS route's specific properties
        t_out_min = route.travel_time_minutes(self.traffic, self.weather)
        t_back_min = (route.distance_km / config.V_SPEED) * self.traffic * \
                     config.WEATHER_STATES[self.weather]['base_mult'] * 60.0

        order.delivery_time  = self.t + t_out_min
        order.delivered      = True
        order.assigned_to    = rider.id
        order.chosen_route   = route_idx

        route_reward = 0.0

        
        if order.is_late():
            rider.total_late += 1

        rider.current_order    = order.id
        rider.status           = 'en_route'
        rider.return_at        = self.t + t_out_min + t_back_min
        rider.total_distance  += route.distance_km * 2   # out + return
        rider.total_deliveries += 1
        rider.route_choices.append(route_idx)
        self.completed += 1

        if (self.weather >= config.SHORTCUT_FLOOD_TRIGGER
                and route.flood_risk > 0.4):
            flood_penalty = config.THETA_4 * route.flood_risk * config.SHORTCUT_FLOOD_MULT
            route_reward -= flood_penalty  # immediate signal

        # Bonus for choosing shortcut in clear weather (it IS fastest)
        if self.weather == 0 and route.route_type == 'shortcut':
            route_reward += config.THETA_4 * 0.3  # small positive signal

        return route_reward



    # ── Objective function J — path-planning focused ──────────────────────────

    def compute_J(self) -> float:
        J = 0.0

        for r in self.riders:
            J += config.C_KM   * r.total_distance
            J += config.THETA_5 * (1.0 - r.health) * r.total_distance

        for o in self.orders.values():
            if o.delivered and o.is_late():
                J += config.P_LATE * o.delay()
            if o.delivered and o.chosen_route is not None:
                chosen = o.routes[o.chosen_route]
                J += chosen.weather_risk_penalty(self.weather)

        for o in self.pending_orders():
            J += config.THETA_1 * o.urgency(self.t)

        return max(0.0, J)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def state(self) -> Dict:
        return {
            't': self.t, 'traffic': self.traffic, 'weather': self.weather,
            'pending_orders': self.pending_orders(),
            'riders': self.riders,
            'n_pending': len(self.pending_orders()),
            'n_idle': len(self.idle_riders()),
        }

    def pending_orders(self) -> List[Order]:
        return [o for o in self.orders.values() if o.assigned_to is None and not o.delivered]

    def idle_riders(self) -> List[Rider]:
        return [r for r in self.riders if r.is_idle]

    def _spawn(self):
        o = sample_order(self._nid, self.t, self.net)
        self.orders[o.id] = o; self._nid += 1

    def _tick(self):
        h = (self.t / 60.0) % 24.0
        for (h0, h1), level in config.TRAFFIC_SCHEDULE.items():
            if h0 <= h < h1:
                self.traffic = config.TRAFFIC_MULTIPLIERS[level]; break
        probs = config.WEATHER_TRANSITION[self.weather]
        self.weather = int(np.random.choice([0, 1, 2], p=config.WEATHER_PROBS))

    def _on_time_rate(self) -> float:
        d = [o for o in self.orders.values() if o.delivered]
        return sum(1 for o in d if not o.is_late()) / max(1, len(d))

    def print_summary(self):
        d   = [o for o in self.orders.values() if o.delivered]
        ot  = [o for o in d if not o.is_late()]
        rc  = [o.chosen_route for o in d if o.chosen_route is not None]
        from collections import Counter
        print('\n' + '='*60)
        print('EPISODE SUMMARY')
        print('='*60)
        print(f'  Delivered : {len(d)}/{len(self.orders)}')
        print(f'  On-time   : {len(ot)} ({len(ot)/max(1,len(d))*100:.1f}%)')
        print(f'  Fleet km  : {sum(r.total_distance for r in self.riders):.2f}')
        print(f'  Cost J    : ₹{self.compute_J():.2f}')
        if rc:
            counts = Counter(rc)
            print(f'  Route usage: {dict(sorted(counts.items()))} (0=arterial,1=residential,2=shortcut)')
        print('='*60)


if __name__ == '__main__':
    env   = QuickCommerceEnv()
    state = env.reset()
    print(f'\nt=0: {state["n_pending"]} orders, {state["n_idle"]} idle riders')
    for step in range(30):
        idle = env.idle_riders()
        decisions = {}
        for r in idle:
            # Simple weather-aware policy: avoid route 2 (shortcut) in rain
            if env.weather >= 2:
                decisions[r.id] = 1   # residential when heavy rain
            elif env.weather == 1:
                decisions[r.id] = 1   # residential when drizzle
            else:
                decisions[r.id] = 2   # shortcut in clear weather
        env.step(decisions)

    env.print_summary()
    print('✓ environment.py OK')