from __future__ import annotations
import math, os, pickle
from typing import Dict, List, Optional, Tuple
import numpy as np
import config


class _StubNetwork:
    ROAD_FACTOR = 1.35

    def _straight(self, a, b):
        dlat = (a[0]-b[0])*111.0
        dlon = (a[1]-b[1])*111.0*math.cos(math.radians((a[0]+b[0])/2))
        return math.hypot(dlat, dlon)

    def road_distance(self, a, b):
        return self._straight(a, b) * self.ROAD_FACTOR

    def route_distance(self, locs):
        return sum(self.road_distance(locs[i], locs[i+1]) for i in range(len(locs)-1))

    def travel_time(self, a, b, traffic, weather_mult, speed=config.V_SPEED):
        return (self.road_distance(a, b) / speed) * traffic * weather_mult

    def road_path(self, a, b):
        return [a, b]

    def k_shortest_distances(self, a, b, k=3):
        base = self.road_distance(a, b)
        return [base * 1.30, base * 1.38, base * 1.18]


class _OSMNetwork:
    def __init__(self, city, cache_path):
        import osmnx as ox, networkx as nx
        self._ox = ox; self._nx = nx; self._cache: Dict = {}

        if os.path.exists(cache_path):
            print(f'  Loading OSM from cache …')
            with open(cache_path, 'rb') as f:
                self.G = pickle.load(f)
        else:
            print(f'  Downloading OSM for {city} …')
            try:
                self.G = ox.graph_from_place(city, network_type='drive', simplify=True)
            except Exception:
                self.G = ox.graph_from_point(
                    config.DARK_STORE_LOCATION,
                    dist=int(config.SERVICE_RADIUS_KM * 1500),
                    network_type='drive', simplify=True)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(self.G, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'  Saved → {cache_path}')

    def _node(self, loc):
        key = (round(loc[0],7), round(loc[1],7))
        if key not in self._cache:
            self._cache[key] = self._ox.distance.nearest_nodes(self.G, loc[1], loc[0])
        return self._cache[key]

    def road_distance(self, a, b):
        na, nb = self._node(a), self._node(b)
        if na == nb: return 0.05
        try:
            return self._nx.shortest_path_length(self.G, na, nb, weight='length') / 1000.0
        except Exception:
            return self._fallback(a, b) * 1.5

    def route_distance(self, locs):
        return sum(self.road_distance(locs[i], locs[i+1]) for i in range(len(locs)-1))

    def travel_time(self, a, b, traffic, weather_mult, speed=config.V_SPEED):
        return (self.road_distance(a, b) / speed) * traffic * weather_mult

    def road_path(self, a, b):
        try:
            na, nb = self._node(a), self._node(b)
            nodes = self._nx.shortest_path(self.G, na, nb, weight='length')
            return [(self.G.nodes[n]['y'], self.G.nodes[n]['x']) for n in nodes]
        except Exception:
            return [a, b]

    def k_shortest_distances(self, a, b, k=3):
        na, nb = self._node(a), self._node(b)
        try:
            paths = list(self._nx.shortest_simple_paths(
                self.G, na, nb, weight='length'))[:k]
            dists = []
            for path in paths:
                d = sum(self.G[path[i]][path[i+1]][0].get('length', 50)
                        for i in range(len(path)-1)) / 1000.0
                dists.append(d)
            while len(dists) < k:
                dists.append(dists[-1] * 1.1)
            return dists
        except Exception:
            base = self.road_distance(a, b)
            return [base * 1.30, base * 1.38, base * 1.18]

    @staticmethod
    def _fallback(a, b):
        dlat = (a[0]-b[0])*111; dlon = (a[1]-b[1])*111*math.cos(math.radians((a[0]+b[0])/2))
        return math.hypot(dlat, dlon)


class RoadNetwork:
    def __init__(self, city=config.CITY, cache_path=config.ROAD_NETWORK_CACHE,
                 use_cache=True):
        try:
            import osmnx
            self._b = _OSMNetwork(city, cache_path)
            n = len(self._b.G.nodes)
            print(f'✓ RoadNetwork: OSM ({n:,} nodes, Bangalore road network)')
        except Exception as e:
            print(f'  OSM unavailable ({e.__class__.__name__}) — using calibrated stub.')
            self._b = _StubNetwork()
            print('✓ RoadNetwork: Stub (Haversine × 1.35, Bangalore-calibrated)')

    def road_distance(self, a, b):           return self._b.road_distance(a, b)
    def route_distance(self, locs):          return self._b.route_distance(locs)
    def travel_time(self, a, b, tr, we, sp=config.V_SPEED):
        return self._b.travel_time(a, b, tr, we, sp)
    def road_path(self, a, b):               return self._b.road_path(a, b)
    def k_shortest_distances(self, a, b, k=config.K_ROUTES):
        return self._b.k_shortest_distances(a, b, k)


if __name__ == '__main__':
    net = RoadNetwork(use_cache=True)
    s   = config.DARK_STORE_LOCATION
    c   = (12.9680, 77.6050)
    d   = net.road_distance(s, c)
    ks  = net.k_shortest_distances(s, c)
    t   = net.travel_time(s, c, 1.3, 1.15) * 60
    print(f'  Shortest: {d:.3f} km')
    print(f'  K-paths:  {[f"{x:.3f}" for x in ks]} km')
    print(f'  Travel:   {t:.1f} min (peak + drizzle)')
    print('✓ road_network.py OK')