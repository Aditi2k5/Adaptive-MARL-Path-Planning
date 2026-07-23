from __future__ import annotations
import argparse
import os
import pickle
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional
import numpy as np
import torch
from tqdm import tqdm
import config
from agent import build_agent_state
from environment import QuickCommerceEnv
from qmix import QMIXCoordinator
from road_network import RoadNetwork

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required. This project is configured to run on CUDA only.")
DEVICE = torch.device("cuda")
print(f"Using device: {DEVICE}")

def policy_random(rider, order, env, network) -> int:
    return random.randint(0, config.K_ROUTES - 1)


def policy_static(rider, order, env, network) -> int:
    """Always Route 0 (arterial main road) — mimics static Dijkstra."""
    return 0


def policy_heuristic(rider, order, env, network) -> int:
    """Best hand-crafted rule: avoid shortcut in rain, use it when clear."""
    return 1 if env.weather >= 1 else 2

def load_coordinator(model_dir, city_name="bangalore", device=None):
    if device is None:
        device = DEVICE
    coord = QMIXCoordinator(n_agents=config.N_RIDERS, device=device)
        # Always created — QMIX always in results table
    candidates = [
            model_dir,
            os.path.join(model_dir, city_name),
            os.path.join("models", city_name, "final"),
            os.path.join("models", "final"),
        ]
    for c in candidates:
            if c and os.path.exists(os.path.join(c, "mixer.pt")):
                coord.load(c)
                return coord, True
    print(f"  No trained model found. Using untrained QMIX.")
    return coord, False
def run_episode(env, network, policy_fn, coordinator) -> Dict:

    env.reset()
    route_by_weather = defaultdict(Counter)

    while env.t < config.EPISODE_LENGTH:
        idle_riders = env.idle_riders()
        pending     = env.pending_orders()

        if not idle_riders or not pending:
            env.step({})
            continue

        pending_sorted = sorted(pending, key=lambda o: o.urgency(env.t), reverse=True)
        pairs = list(zip(idle_riders, pending_sorted))

        if coordinator is not None:
            # Build ONE observation per rider (all N_RIDERS, padded with zeros)
            rider_order_map = {r.id: o for r, o in pairs}
            obs_list = []
            for rider in env.riders:                   # ALL riders, not just idle
                order = rider_order_map.get(rider.id)
                if order is not None and rider.is_idle:
                    obs = build_agent_state(
                        rider, order, env.traffic, env.weather, env.t, network
                    )
                else:
                    obs = np.zeros(config.STATE_DIM, dtype=np.float32)
                obs_list.append(obs)

            # obs_list is now exactly N_RIDERS long
            all_actions = coordinator.select_actions(obs_list, epsilon=0.0)

            path_decisions = {}
            for rider, order in pairs:
                route_idx = all_actions[rider.id]
                path_decisions[rider.id] = route_idx
                route_by_weather[env.weather][route_idx] += 1
        else:
            path_decisions = {}
            for rider, order in pairs:
                route_idx = policy_fn(rider, order, env, network)
                path_decisions[rider.id] = route_idx
                route_by_weather[env.weather][route_idx] += 1

        _, _, done, _ = env.step(path_decisions)
        if done:
            break

    delivered = [o for o in env.orders.values() if o.delivered]
    on_time   = [o for o in delivered if not o.is_late()]
    late      = [o for o in delivered if o.is_late()]

    return {
        "cost_J":           env.compute_J(),
        "on_time_rate":     len(on_time) / max(1, len(delivered)),
        "total_dist_km":    sum(r.total_distance for r in env.riders),
        "n_delivered":      len(delivered),
        "n_late":           len(late),
        "avg_delay_min":    float(np.mean([o.delay() for o in late])) if late else 0.0,
        "route_by_weather": {k: dict(v) for k, v in route_by_weather.items()},
    }

def evaluate(city_name: str = "bangalore", model_dir: Optional[str] = None, n_episodes: int = 100) -> Dict:

    # Load city-specific configuration
    config.load_city(city_name)

    network              = RoadNetwork(use_cache=True)
    env                  = QuickCommerceEnv(network=network)
    coordinator, trained = load_coordinator(model_dir, city_name=city_name, device=DEVICE)
    qmix_label           = "QMIX MARL (Trained)" if trained else "QMIX MARL (Untrained*)"

    # All 4 strategies — QMIX always present
    strategies = {
        "Random Route":            (policy_random,    None),
        "Static Dijkstra":         (policy_static,    None),
        "Weather-Aware Heuristic": (policy_heuristic, None),
        qmix_label:                (None, coordinator),   # ALWAYS added
    }

    print(f"\nEvaluating {len(strategies)} strategies × {n_episodes} episodes each\n")

    results = {}
    for name, (pol_fn, coord) in strategies.items():
        costs=[]; on_times=[]; dists=[]; delays=[]
        rw_agg = defaultdict(Counter)

        for _ in tqdm(range(n_episodes), desc=f"  {name:<32}", leave=False):
            m = run_episode(env, network, pol_fn, coord)
            costs.append(m["cost_J"])
            on_times.append(m["on_time_rate"])
            dists.append(m["total_dist_km"])
            delays.append(m["avg_delay_min"])
            for ws, rc in m["route_by_weather"].items():
                for ri, cnt in rc.items():
                    rw_agg[ws][ri] += cnt
            results[name]["all_J_values"] = costs

        results[name] = {
            "mean_cost":        float(np.mean(costs)),
            "std_cost":         float(np.std(costs)),
            "mean_on_time":     float(np.mean(on_times)) * 100,
            "mean_dist":        float(np.mean(dists)),
            "mean_delay":       float(np.mean(delays)),
            "route_by_weather": {k: dict(v) for k, v in rw_agg.items()},
        }

    print_results(results, trained)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out = os.path.join(config.RESULTS_DIR, "eval_results.pkl")
    with open(out, "wb") as f:
        pickle.dump(results, f)
    print(f"\n  Results saved → {out}")
    return results

def print_results(results: Dict, is_trained: bool) -> None:
    names    = list(results.keys())
    baseline = results[names[0]]["mean_cost"]

    print("\n" + "=" * 84)
    print("EVALUATION RESULTS  —  Adaptive Path Planning, Quick Commerce")
    print("=" * 84)
    print(f"  {'Strategy':<35} {'J (RS)':>8} {'+-':>6} "
          f"{'OnTime%':>8} {'Dist km':>8} {'vs Random':>10}")
    print("  " + "-" * 80)

    for name, m in results.items():
        imp  = (baseline - m["mean_cost"]) / baseline * 100
        mark = "  <-- OURS" if "QMIX" in name else ""
        print(f"  {name:<35} {m['mean_cost']:>7.2f} {m['std_cost']:>7.2f}"
              f" {m['mean_on_time']:>7.1f}% {m['mean_dist']:>8.2f}"
              f" {imp:>+9.1f}%{mark}")

    print("=" * 84)

    if not is_trained:
        print()
        print("  * QMIX Untrained = random weights. "
              "Run: python train.py --episodes 3000")
        print("    Then: python evaluate.py --model_dir models/final")

    # Route usage per weather condition
    print("\n\nROUTE USAGE BY WEATHER CONDITION")
    print("─" * 84)
    weather_labels = {0: "Clear (w=0)", 1: "Drizzle (w=1)", 2: "Heavy Rain (w=2)"}

    for ws in [0, 1, 2]:
        print(f"\n  {weather_labels[ws]}:")
        print(f"  {'Strategy':<35} {'R0 Arterial':>13} "
              f"{'R1 Residential':>15} {'R2 Shortcut':>13}")
        print("  " + "-" * 78)
        for name, m in results.items():
            rw  = m["route_by_weather"].get(ws, {})
            tot = sum(rw.values()) + 1e-9
            r0  = rw.get(0, 0) / tot * 100
            r1  = rw.get(1, 0) / tot * 100
            r2  = rw.get(2, 0) / tot * 100
            print(f"  {name:<35} {r0:>11.0f}%  {r1:>13.0f}%  {r2:>11.0f}%")

    print("\n  KEY insight (after training):")
    print("  Heavy Rain → QMIX should use R2 (shortcut) near 0% (avoids flood)")
    print("  Clear      → QMIX should use R2 highly (fastest when dry)")
    print("  Static Dijkstra always uses R2=100% regardless — cannot adapt")

    # Summary box
    if len(names) >= 4:
        r = results; rnd = r[names[0]]; heu = r[names[2]]; qmx = r[names[3]]
        print("\n" + "=" * 84)
        print("SUMMARY")
        print("=" * 84)
        print(f"  Random     J=RS{rnd['mean_cost']:7.2f}  on-time={rnd['mean_on_time']:.1f}%")
        print(f"  Heuristic  J=RS{heu['mean_cost']:7.2f}  on-time={heu['mean_on_time']:.1f}%"
              f"  ({(rnd['mean_cost']-heu['mean_cost'])/rnd['mean_cost']*100:+.1f}%)")
        print(f"  QMIX       J=RS{qmx['mean_cost']:7.2f}  on-time={qmx['mean_on_time']:.1f}%"
              f"  ({(rnd['mean_cost']-qmx['mean_cost'])/rnd['mean_cost']*100:+.1f}%)")
        vs = (heu['mean_cost'] - qmx['mean_cost']) / heu['mean_cost'] * 100
        print(f"  QMIX vs Heuristic: {vs:+.1f}%  "
              f"({'beats heuristic' if vs > 0 else 'needs more training'})")
        print("=" * 84)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate adaptive path planning")
    ap.add_argument("--model_dir", default=None,
                    help="QMIX checkpoint dir (e.g. models/final)")
    ap.add_argument("--episodes",  type=int, default=100)
    ap.add_argument("--city", type=str, default="bangalore")
    args = ap.parse_args()
    evaluate(city_name=args.city, model_dir=args.model_dir, n_episodes=args.episodes)
