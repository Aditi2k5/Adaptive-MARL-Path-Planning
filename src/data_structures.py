from __future__ import annotations
import math, random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from src import config


@dataclass
class Route:
    id:           int
    route_type:   str         # 'arterial' | 'residential' | 'shortcut'
    distance_km:  float       # actual road distance
    flood_risk:   float       # 0.0 = never floods, 1.0 = always floods
    waypoints:    List[Tuple[float, float]] = field(default_factory=list)

    def travel_time(self, base_traffic, weather_state, speed=None):
        from src import config
        if speed is None:
            speed = config.V_SPEED
        props_traffic = config.ROUTE_TRAFFIC_MULTS[self.route_type]
        props_weather = config.ROUTE_WEATHER_MULTS[self.route_type]
        eff_traffic = base_traffic * props_traffic
        base_weather = config.WEATHER_MULTS[weather_state]
        eff_weather  = base_weather * props_weather
        # City-specific flood trigger and multiplier
        if (weather_state >= config.SHORTCUT_FLOOD_TRIGGER
                and self.flood_risk > 0.4):
            eff_weather *= config.SHORTCUT_FLOOD_MULT
        return (self.distance_km / speed) * eff_traffic * eff_weather
    
    def travel_time_minutes(self, base_traffic: float,
                            weather_state: int,
                            speed: float = config.V_SPEED) -> float:
        return self.travel_time(base_traffic, weather_state, speed) * 60.0

    def weather_risk_penalty(self, weather_state: int) -> float:
        """θ₄ × risk_level for this route under current weather."""
        w_risk = config.WEATHER_STATES[weather_state]['risk']
        route_risk = w_risk * self.flood_risk
        return config.THETA_4 * route_risk

    def breakdown_risk(self, health: float) -> float:
        """θ₅ × (1−h) × distance — longer routes more dangerous for unhealthy bikes."""
        return config.THETA_5 * (1.0 - health) * self.distance_km

@dataclass
class Order:
    id:                int
    location:          Tuple[float, float]
    arrival_time:      float
    deadline_duration: int
    routes:            List[Route] = field(default_factory=list)  # K_ROUTES alternatives

    assigned_to:   Optional[int]   = field(default=None)
    chosen_route:  Optional[int]   = field(default=None)   # which route the agent picked
    delivered:     bool            = False
    delivery_time: Optional[float] = None
    deadline_time: float           = field(init=False)

    def __post_init__(self):
        self.deadline_time = self.arrival_time + self.deadline_duration

    def urgency(self, t: float) -> float:
        """u_i(t) = 1 − (t_i^d − t) / d_i  (paper formula)."""
        return float(np.clip(
            1.0 - (self.deadline_time - t) / self.deadline_duration, 0.0, 1.0
        ))

    def is_late(self) -> bool:
        return self.delivery_time is not None and self.delivery_time > self.deadline_time

    def delay(self) -> float:
        return max(0.0, (self.delivery_time or 0.0) - self.deadline_time)

@dataclass
class Rider:
    id:       int
    location: Tuple[float, float]
    health:   float = 1.0         # h_j ∈ [0,1]
    speed:    float = config.V_SPEED

    status:         str            = 'idle'
    current_order:  Optional[int]  = None
    return_at:      float          = 0.0    # when rider gets back to store

    total_distance:   float = 0.0
    total_deliveries: int   = 0
    total_late:       int   = 0
    route_choices:    List[int] = field(default_factory=list)  # log of routes chosen

    @property
    def is_idle(self) -> bool:
        return self.status == 'idle'

    def reset(self) -> None:
        self.location         = config.DARK_STORE_LOCATION
        self.status           = 'idle'
        self.current_order    = None
        self.return_at        = 0.0
        self.total_distance   = 0.0    # RESET each episode
        self.total_deliveries = 0
        self.total_late       = 0
        self.route_choices    = []

def generate_route_alternatives(store, customer, network):
        """
        Generate K_ROUTES=3 route alternatives using city-specific parameters.
        Uses config.ROUTE_DIST_FACTORS, ROUTE_FLOOD_RISKS etc.
        These are set by config.load_city() so this works for all cities.
        """
        from src import config
        base_dist = network.road_distance(store, customer)
 
        routes = []
        for i, rtype in enumerate(["arterial", "residential", "shortcut"]):
            dist = base_dist * config.ROUTE_DIST_FACTORS[rtype]
            flood = config.ROUTE_FLOOD_RISKS[rtype]
            routes.append(Route(
                id=i,
                route_type=rtype,
                distance_km=dist,
                flood_risk=flood,
                waypoints=[store, customer],  # OSM fills real waypoints
            ))
        return routes

def sample_order(order_id: int, current_time: float, network) -> Order:
    """Generate one order within 3 km, with 3 route alternatives attached."""
    store_lat, store_lon = config.DARK_STORE_LOCATION
    angle    = random.uniform(0, 2 * math.pi)
    dist_km  = random.uniform(0.3, config.SERVICE_RADIUS_KM)

    location = (
        store_lat + (dist_km / 111.0) * math.cos(angle),
        store_lon + (dist_km / (111.0 * math.cos(math.radians(store_lat)))) * math.sin(angle),
    )

    # Deadline: must be achievable for single delivery on worst route
    road_km    = dist_km * 1.35
    t_worst    = (road_km * 1.38 / config.V_SPEED) * 1.3 * 1.15 * 60  # residential, peak+drizzle
    d_i = random.choices(config.DEADLINE_DURATIONS, weights=config.DEADLINE_WEIGHTS, k=1)[0]
    if t_worst > d_i * 0.85:
        d_i = 30

    order = Order(
        id=order_id,
        location=location,
        arrival_time=current_time,
        deadline_duration=d_i,
    )
    order.routes = generate_route_alternatives(
        config.DARK_STORE_LOCATION, location, network
    )
    return order


def make_fleet(n: int = config.N_RIDERS) -> List[Rider]:
    return [
        Rider(id=j, location=config.DARK_STORE_LOCATION,
              health=round(random.uniform(0.75, 1.0), 2))
        for j in range(n)
    ]

if __name__ == '__main__':
    from road_network import RoadNetwork
    random.seed(42); np.random.seed(42)

    print('data_structures.py — self-test\n')
    net   = RoadNetwork(use_cache=True)
    order = sample_order(0, 0.0, net)

    print(f'Order 0: location={order.location}')
    print(f'  deadline={order.deadline_duration}min  urgency@t=10: {order.urgency(10):.3f}')
    print(f'  {len(order.routes)} route alternatives:')

    conditions = [(1.0,0,'Normal'), (1.3,0,'Peak'), (1.3,2,'Peak+Rain')]
    print(f"  {'Route':<18} {'Type':<14} {'Dist':>6} {'Norm':>7} {'Peak':>7} {'Peak+Rain':>10}")
    print('  ' + '-'*65)
    for r in order.routes:
        times = [r.travel_time_minutes(tr, ws) for tr,ws,_ in conditions]
        print(f"  Route {r.id} ({r.route_type:<12}) {r.distance_km:>5.2f}km "
              f"  {times[0]:>5.1f}m  {times[1]:>5.1f}m  {times[2]:>7.1f}m"
              f"  flood_risk={r.flood_risk:.2f}")

    print('\nRoute selection analysis:')
    for tr,ws,name in conditions:
        best = min(range(3), key=lambda i: order.routes[i].travel_time(tr,ws))
        print(f'  {name}: best route = Route {best} ({order.routes[best].route_type})')

    print('\n✓ data_structures.py OK')