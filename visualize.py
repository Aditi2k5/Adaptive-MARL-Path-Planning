from __future__ import annotations
import os
import pickle
from typing import Dict, List, Optional
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import config
RESULTS_DIR = config.RESULTS_DIR
os.makedirs(RESULTS_DIR, exist_ok=True)

def _moving_avg(x: List[float], w: int = 50) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def _save(fig: plt.Figure, name: str) -> str:
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ saved → {path}")
    return path


def _pick_series(data: Dict, *names: str):
    for name in names:
        if name in data:
            return data[name]
    raise KeyError(f"None of the expected keys found: {', '.join(names)}")


def _series_or_scalar(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float)
        return float(np.mean(arr)), float(np.std(arr))
    return float(value), 0.0


def plot_learning_curves(log_path: str = None,
                         log: Dict = None,
                         window: int = 50) -> None:
    """
    Plot CostJ, On-time rate, and Loss over training episodes.
    Accepts either a path to a .pkl log file or the dict directly.
    """
    if log is None:
        if log_path is None:
            log_path = os.path.join(config.RESULTS_DIR, "training_log.pkl")
        with open(log_path, "rb") as f:
            log = pickle.load(f)

    costs    = _pick_series(log, "episode_costs", "costs")
    on_times = _pick_series(log, "on_time_rates", "on_times")
    losses   = _pick_series(log, "losses")
    episodes = np.arange(1, len(costs) + 1)
    window = max(1, min(window, len(costs)))

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle("QMIX Training Progress – Quick Commerce MARL", fontsize=14)

    ax = axes[0]
    ax.plot(episodes, costs, alpha=0.25, color="steelblue", lw=0.8)
    ax.plot(episodes[window - 1:],
            _moving_avg(costs, window),
            color="steelblue", lw=2.0, label=f"{window}-ep MA")
    ax.set_ylabel("Cost J  (₹)")
    ax.set_title("Objective Function J(x, π)  ↓ lower is better")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    on_times_pct = np.array(on_times) * 100
    ax.plot(episodes, on_times_pct, alpha=0.25, color="seagreen", lw=0.8)
    ax.plot(episodes[window - 1:],
            _moving_avg(on_times_pct, window),
            color="seagreen", lw=2.0, label=f"{window}-ep MA")
    ax.axhline(90, color="red", ls="--", lw=1.2, label="90% target")
    ax.set_ylabel("On-time rate  (%)")
    ax.set_title("On-time Delivery Rate  ↑ higher is better")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(episodes, losses, alpha=0.25, color="coral", lw=0.8)
    ax.plot(episodes[window - 1:],
            _moving_avg(losses, window),
            color="coral", lw=2.0, label=f"{window}-ep MA")
    ax.set_ylabel("TD Loss")
    ax.set_xlabel("Episode")
    ax.set_title("QMIX TD Loss  ↓ lower is better")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, "learning_curves.png")

def plot_comparison(results: Dict[str, Dict] = None,
                    results_path: str = None) -> None:

    if results is None:
        if results_path is None:
            results_path = os.path.join(config.RESULTS_DIR, "eval_results.pkl")
        with open(results_path, "rb") as f:
            results = pickle.load(f)

    names   = list(results.keys())
    costs = []
    on_times = []
    costs_err = []
    on_err = []
    for name in names:
        cost_mean, cost_std = _series_or_scalar(_pick_series(results[name], "costs", "mean_cost"))
        on_mean, on_std = _series_or_scalar(_pick_series(results[name], "on_time", "on_time_rate", "mean_on_time"))
        costs.append(cost_mean)
        costs_err.append(cost_std)
        on_times.append(on_mean * 100 if on_mean <= 1.0 else on_mean)
        on_err.append(on_std * 100 if on_std <= 1.0 else on_std)

    colours = ["#5b9bd5"] * (len(names) - 1) + ["#e84c3d"]
    if "QMIX (Ours)" not in names:
        colours = ["#5b9bd5"] * len(names)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Strategy Comparison – Quick Commerce Delivery", fontsize=13)

    x = np.arange(len(names))
    w = 0.55

    bars = ax1.bar(x, costs, w, yerr=costs_err, capsize=4,
                   color=colours, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=18, ha="right")
    ax1.set_ylabel("Mean Cost J  (₹)")
    ax1.set_title("Objective Function J(x, π)  ↓")
    ax1.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, costs):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(costs) * 0.01,
                 f"₹{v:.1f}", ha="center", va="bottom", fontsize=8)

    # On-time rate
    bars2 = ax2.bar(x, on_times, w, yerr=on_err, capsize=4,
                    color=colours, edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=18, ha="right")
    ax2.set_ylabel("On-time delivery rate  (%)")
    ax2.set_title("On-time Rate  ↑")
    ax2.set_ylim(0, 105)
    ax2.axhline(90, color="gray", ls="--", lw=1.0)
    ax2.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars2, on_times):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    # Legend for colour coding
    patch_base = mpatches.Patch(color="#5b9bd5", label="Baseline")
    patch_ours = mpatches.Patch(color="#e84c3d", label="QMIX (Ours)")
    fig.legend(handles=[patch_base, patch_ours], loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    _save(fig, "comparison_chart.png")


def plot_routes_on_map(routes_data: List[Dict],
                       title: str = "Delivery Routes") -> None:

    try:
        import osmnx as ox
        import networkx as nx
        from road_network import RoadNetwork

        net = RoadNetwork(use_cache=True)

        fig, ax = ox.plot_graph(net.G,
                                figsize=(12, 10),
                                show=False,
                                close=False,
                                node_size=0,
                                edge_linewidth=0.5,
                                edge_color="#aaaaaa",
                                bgcolor="white")
    except Exception:
        # Fallback: plain matplotlib axes
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_facecolor("#f0f0f0")

    ax.set_title(title, fontsize=13)

    for route in routes_data:
        wps    = route["waypoints"]
        color  = route.get("color", "blue")
        label  = route.get("label", "")
        lats   = [p[0] for p in wps]
        lons   = [p[1] for p in wps]
        ax.plot(lons, lats, "-o", color=color, lw=2.0,
                markersize=4, label=label, alpha=0.85)

    # Mark dark store
    store_lat, store_lon = config.DARK_STORE_LOCATION
    ax.plot(store_lon, store_lat, "s", color="black",
            markersize=12, zorder=5, label="Dark Store")

    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, "routes_map.png")


def plot_epsilon(log: Dict = None, log_path: str = None) -> None:
    if log is None:
        with open(log_path or os.path.join(config.RESULTS_DIR,
                                           "training_log.pkl"), "rb") as f:
            log = pickle.load(f)
    eps = log["epsilons"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(eps, color="purple", lw=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon  ε")
    ax.set_title("Exploration Rate Decay")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "epsilon_decay.png")


def generate_all(log_path: str = None,
                 results_path: str = None) -> None:
    """Generate every plot from saved log files."""
    print("\nGenerating all visualisations …")

    log_p = log_path     or os.path.join(config.RESULTS_DIR, "training_log.pkl")
    res_p = results_path or os.path.join(config.RESULTS_DIR, "eval_results.pkl")

    if os.path.exists(log_p):
        plot_learning_curves(log_path=log_p)
        plot_epsilon(log_path=log_p)
    else:
        print(f"  ⚠  Training log not found at {log_p} – skipping learning curves.")

    if os.path.exists(res_p):
        plot_comparison(results_path=res_p)
    else:
        print(f"  ⚠  Eval results not found at {res_p} – skipping comparison chart.")

    print("\n✓ All available plots saved to", RESULTS_DIR)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--log",     default=None, help="Path to training_log.pkl")
    parser.add_argument("--results", default=None, help="Path to eval_results.pkl")
    parser.add_argument("--plot_routes_on_map", action="store_true",
                        help="Plot delivery routes on OSM map (uses --routes if given)")
    parser.add_argument("--routes", default=None,
                        help="Path to a pickle file containing routes_data (list of dicts)")
    args = parser.parse_args()

    if args.plot_routes_on_map:
        # Load routes if provided, otherwise generate a small demo around the dark store
        routes = None
        if args.routes:
            with open(args.routes, "rb") as f:
                routes = pickle.load(f)
        else:
            from road_network import RoadNetwork
            net = RoadNetwork(use_cache=True)
            ds = config.DARK_STORE_LOCATION
            # create three sample customer points around the dark store
            deltas = [(0.012, 0.006), (-0.010, 0.008), (0.007, -0.011)]
            demo_targets = [(ds[0] + d[0], ds[1] + d[1]) for d in deltas]
            routes = []
            cols = ["#2b83ba", "#abdda4", "#fdae61"]
            for i, tgt in enumerate(demo_targets):
                wps = net.road_path(ds, tgt)
                routes.append({"waypoints": wps, "color": cols[i % len(cols)], "label": f"Demo Route {i}"})

        plot_routes_on_map(routes, title="Demo Delivery Routes")
        sys.exit(0)

    generate_all(log_path=args.log, results_path=args.results)