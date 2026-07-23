from __future__ import annotations
import itertools
import math
import random
from typing import Dict, List, Tuple
import config
from data_structures import Order, Rider

def random_assign(orders: List[Order],
                  riders: List[Rider]) -> Dict[int, int]:
    """Random assignment baseline."""
    assignments: Dict[int, int] = {}
    available = [r for r in riders if r.has_capacity()]
    for order in orders:
        if not available:
            break
        rider = random.choice(available)
        assignments[order.id] = rider.id
        if len([v for v in assignments.values() if v == rider.id]) >= rider.capacity:
            available.remove(rider)
    return assignments


def greedy_urgent_assign(orders: List[Order],
                         riders: List[Rider],
                         current_time: float) -> Dict[int, int]:

    assignments: Dict[int, int] = {}
    # Sort most urgent first
    sorted_orders = sorted(orders,
                           key=lambda o: o.urgency(current_time),
                           reverse=True)
    available = [r for r in riders if r.has_capacity()]

    for order in sorted_orders:
        if not available:
            break
        rider = available[0]
        assignments[order.id] = rider.id
        assigned_to_rider = [v for v in assignments.values() if v == rider.id]
        if len(assigned_to_rider) >= rider.capacity:
            available.pop(0)

    return assignments


def nearest_rider_assign(orders: List[Order],
                         riders: List[Rider],
                         network) -> Dict[int, int]:
    """Assign each order to the nearest available rider."""
    assignments: Dict[int, int] = {}
    for order in orders:
        best_rid, best_dist = None, float("inf")
        for rider in riders:
            if rider.has_capacity():
                taken = len([v for v in assignments.values() if v == rider.id])
                if taken < rider.capacity:
                    d = network.road_distance(rider.location, order.location)
                    if d < best_dist:
                        best_dist, best_rid = d, rider.id
        if best_rid is not None:
            assignments[order.id] = best_rid
    return assignments

def random_route(order_ids: List[int]) -> List[int]:
    """Random permutation of order_ids."""
    r = order_ids.copy()
    random.shuffle(r)
    return r


def greedy_nearest_route(order_ids: List[int],
                         orders: Dict[int, Order],
                         start_location: Tuple[float, float],
                         network) -> List[int]:
        
    if not order_ids:
        return []

    remaining = order_ids.copy()
    route     = []
    cur_loc   = start_location

    while remaining:
        nearest = min(remaining,
                      key=lambda oid: network.road_distance(cur_loc,
                                                            orders[oid].location))
        route.append(nearest)
        cur_loc = orders[nearest].location
        remaining.remove(nearest)

    return route


def urgent_first_route(order_ids: List[int],
                       orders: Dict[int, Order],
                       current_time: float) -> List[int]:
    return sorted(order_ids,
                  key=lambda oid: orders[oid].urgency(current_time),
                  reverse=True)


def optimal_route_brute(order_ids: List[int],
                        orders: Dict[int, Order],
                        start_location: Tuple[float, float],
                        network) -> List[int]:
    if not order_ids:
        return []
    if len(order_ids) == 1:
        return order_ids

    best_route, best_dist = order_ids, float("inf")

    for perm in itertools.permutations(order_ids):
        locations = [start_location] + [orders[oid].location for oid in perm]
        dist = network.route_distance(locations)
        if dist < best_dist:
            best_dist  = dist
            best_route = list(perm)

    return best_route


def action_index_to_route(action: int, order_ids: List[int]) -> List[int]:
    perms = list(itertools.permutations(order_ids))
    if action >= len(perms):
        action = 0
    return list(perms[action])


def route_to_action_index(route: List[int], order_ids: List[int]) -> int:
    """Inverse of action_index_to_route."""
    perms = list(itertools.permutations(order_ids))
    try:
        return perms.index(tuple(route))
    except ValueError:
        return 0


def n_actions(n_orders: int) -> int:
    return max(1, min(math.factorial(n_orders), config.ACTION_DIM))

if __name__ == "__main__":
    from data_structures import sample_order, make_fleet
    from road_network import RoadNetwork

    print("assignment.py  –  self-test\n")

    net    = RoadNetwork(use_cache=True)
    orders = [sample_order(i, 0.0) for i in range(5)]
    riders = make_fleet(3)
    o_dict = {o.id: o for o in orders}

    print("Greedy urgent assignment:")
    asgn = greedy_urgent_assign(orders, riders, current_time=3.0)
    for oid, rid in asgn.items():
        u = orders[oid].urgency(3.0)
        print(f"  order {oid} (u={u:.2f}, d={orders[oid].deadline_duration}min) "
              f"→ rider {rid}")

    print("\nGreedy nearest route for orders 0,1,2:")
    r = greedy_nearest_route([0, 1, 2], o_dict,
                             config.DARK_STORE_LOCATION, net)
    print(f"  {r}")

    print("\nOptimal brute-force route (≤3 orders):")
    r_opt = optimal_route_brute([0, 1, 2], o_dict,
                                config.DARK_STORE_LOCATION, net)
    print(f"  {r_opt}")

    print("\n✓ assignment.py OK")