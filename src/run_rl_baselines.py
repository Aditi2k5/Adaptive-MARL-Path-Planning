"""
run_rl_baselines.py
Trains and evaluates all RL baselines on Bangalore.
Produces Table comparing: DQN, IQL, VDN, QMIX, MAPPO
Run: python run_rl_baselines.py --city bangalore --episodes 3000
"""

import argparse, logging, os, pickle
import numpy as np
import torch
from tqdm import tqdm
import config
from rl_baselines import DQNAgent, IQLCoordinator, VDNCoordinator, MAPPOCoordinator
from qmix import QMIXCoordinator, ReplayBuffer
from agent import build_agent_state, build_global_state
from environment import QuickCommerceEnv
from road_network import RoadNetwork


LOGGER = logging.getLogger("rl_baselines_runner")


def _require_cuda(device: torch.device = None) -> torch.device:
    if device is not None:
        return device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. This runner is configured to run on CUDA only.")
    return torch.device("cuda")


def _configure_logger(city):
    os.makedirs("results", exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False

    if LOGGER.handlers:
        for handler in list(LOGGER.handlers):
            LOGGER.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(os.path.join("results", f"rl_baselines_{city}.log"), mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(console_handler)
    LOGGER.addHandler(file_handler)
    return LOGGER


def _log(message):
    LOGGER.info(message)


def _device_banner(device: torch.device) -> str:
    if device.type != "cuda":
        return str(device)
    idx = torch.cuda.current_device()
    return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"


def _action_for_rider(actions, rider_id, default=0):
    """Return a route decision for one rider from scalar, list, or dict actions."""
    if isinstance(actions, dict):
        return int(actions.get(rider_id, default))
    if isinstance(actions, (list, tuple, np.ndarray)):
        if rider_id < len(actions):
            return int(actions[rider_id])
        return int(default)
    return int(actions)
 
def run_episode_with_method(env, network, method_name, coordinator, epsilon=0.0,
                           phase="train", episode_idx=None, log_steps=False):
    """Run one episode with a given method."""
    env.reset()
    route_counts={0:0,1:0,2:0}
    step_idx = 0

    if episode_idx is None:
        episode_tag = f"{phase}"
    else:
        episode_tag = f"{phase} ep{episode_idx}"

    if log_steps:
        _log(f"[{method_name} | {episode_tag}] episode started")
 
    while env.t < config.EPISODE_LENGTH:
        step_idx += 1
        idle   = env.idle_riders()
        pending= env.pending_orders()
        if not idle or not pending:
            if log_steps:
                _log(f"[{method_name} | {episode_tag}] step {step_idx:04d} t={env.t:03.0f} idle={len(idle)} pending={len(pending)} -> wait")
            env.step({}); continue
 
        pending_sorted = sorted(pending, key=lambda o: o.urgency(env.t), reverse=True)
        pairs = list(zip(idle, pending_sorted))
        rider_order_map = {r.id: o for r,o in pairs}
 
        # Build obs for all riders
        obs_list = []
        for rider in env.riders:
            order = rider_order_map.get(rider.id)
            if order is not None and rider.is_idle:
                obs_list.append(build_agent_state(rider, order, env.traffic, env.weather, env.t, network))
            else:
                import numpy as np
                obs_list.append(np.zeros(config.STATE_DIM, dtype=np.float32))
 
        actions = coordinator.act(obs_list, epsilon=epsilon)
 
        path_decisions = {}
        for rider, order in pairs:
            a = _action_for_rider(actions, rider.id)
            path_decisions[rider.id] = a
            route_counts[a] = route_counts.get(a, 0) + 1

        if log_steps:
            assignments = ", ".join(f"r{rider.id}->a{path_decisions[rider.id]}" for rider, _ in pairs)
            _log(f"[{method_name} | {episode_tag}] step {step_idx:04d} t={env.t:03.0f} idle={len(idle)} pending={len(pending)} {assignments}")
 
        _, reward, done, info = env.step(path_decisions)
        if log_steps:
            _log(f"[{method_name} | {episode_tag}] step {step_idx:04d} reward={reward:.3f} J={info['cost_J']:.2f} completed={info['completed']} on_time={info['on_time_rate']*100:.1f}%")
        if done: break
 
    delivered = [o for o in env.orders.values() if o.delivered]
    on_time   = sum(1 for o in delivered if not o.is_late())
    if log_steps:
        _log(f"[{method_name} | {episode_tag}] episode finished steps={step_idx} J={env.compute_J():.2f} on_time={on_time / max(1, len(delivered)) * 100:.1f}%")
    return {
        "J":         env.compute_J(),
        "on_time":   on_time / max(1, len(delivered)),
        "dist":      sum(r.total_distance for r in env.riders),
        "routes":    route_counts,
        "steps":     step_idx,
    }
 
 
def train_method(method_name, episodes, env, network, device, log_steps=False, progress_every=25):
 
    if method_name == "DQN":
        coord = DQNAgent(device=device)
    elif method_name == "IQL":
        coord = IQLCoordinator(device=device)
    elif method_name == "VDN":
        coord = VDNCoordinator(device=device)
    elif method_name == "MAPPO":
        coord = MAPPOCoordinator(device=device)
    elif method_name == "QMIX":
        coord = QMIXCoordinator(device=device)
    else:
        raise ValueError(f"Unknown method: {method_name}")
 
    _log(f"Training {method_name} ({episodes} episodes) on {_device_banner(device)}...")
    AJ=[]; AOT=[]
    eps = config.EPSILON_START
 
    with tqdm(total=episodes, desc=f"{method_name} train", unit="ep", leave=False) as bar:
        for ep in range(1, episodes+1):
            result = run_episode_with_method(
                env, network, method_name, coord, epsilon=eps,
                phase="train", episode_idx=ep, log_steps=log_steps
            )
            AJ.append(result["J"]); AOT.append(result["on_time"])
            eps = max(config.EPSILON_END, eps * config.EPSILON_DECAY)
            if hasattr(coord, 'sync_targets') and ep % config.TARGET_UPDATE_FREQ == 0:
                coord.sync_targets()
            if hasattr(coord, 'decay_epsilon'):
                coord.decay_epsilon()

            if ep % progress_every == 0 or ep == 1 or ep == episodes:
                _log(f"[{method_name} | train ep{ep}/{episodes}] J={result['J']:.2f} on_time={result['on_time']*100:.1f}% eps={eps:.4f}")

            bar.set_postfix(J=f"{result['J']:.1f}", OT=f"{result['on_time']*100:.1f}%", eps=f"{eps:.4f}")
            bar.update(1)
 
    return coord
 
 
def evaluate_all(city, episodes=3000, eval_ep=100, quick_test=False, methods=None, log_steps=False):
    _configure_logger(city)
    device = _require_cuda()
    _log(f"Using CUDA device: {_device_banner(device)}")
    _log(f"Preparing {city} city config and road network...")
    config.load_city(city)
    network = RoadNetwork(use_cache=True)
    env     = QuickCommerceEnv(network=network)
 
    if methods is None:
        methods = ["DQN", "IQL", "VDN", "MAPPO", "QMIX"]
    if quick_test:
        episodes = min(episodes, 1)
        eval_ep = min(eval_ep, 1)
        methods = methods[:1]
        _log("Quick test mode enabled: running one method, one training episode, one eval episode.")

    results = {}
 
    for method in methods:
        coord = train_method(method, episodes, env, network, device=device, log_steps=log_steps)
 
        _log(f"Evaluating {method} ({eval_ep} episodes)...")
        Js=[]; OTs=[]; DIs=[]
        with tqdm(total=eval_ep, desc=f"{method} eval", unit="ep", leave=False) as bar:
            for eval_idx in range(1, eval_ep+1):
                r = run_episode_with_method(
                    env, network, method, coord, epsilon=0.0,
                    phase="eval", episode_idx=eval_idx, log_steps=False
                )
                Js.append(r["J"]); OTs.append(r["on_time"]); DIs.append(r["dist"])
                if eval_idx % 10 == 0 or eval_idx == 1 or eval_idx == eval_ep:
                    _log(f"[{method} | eval ep{eval_idx}/{eval_ep}] J={r['J']:.2f} on_time={r['on_time']*100:.1f}%")
                bar.set_postfix(J=f"{r['J']:.1f}", OT=f"{r['on_time']*100:.1f}%")
                bar.update(1)
 
        results[method] = {
            "mean_J":  np.mean(Js),  "std_J":  np.std(Js),
            "mean_OT": np.mean(OTs)*100, "std_OT": np.std(OTs)*100,
            "all_Js":  Js,           "all_OTs": OTs,
        }
 
    # Print table
    _log(f"{'='*70}")
    _log(f"RL BASELINE COMPARISON — {city.upper()}")
    _log(f"{'='*70}")
    has_qmix = "QMIX" in results
    if has_qmix:
        _log(f"{'Method':<12} {'J (mean)':>10} {'J (std)':>8} {'OnTime%':>9} {'vs QMIX J':>10}")
        _log("-"*55)
        qmix_J = results["QMIX"]["mean_J"]
        for m, r in results.items():
            diff = (qmix_J - r["mean_J"]) / qmix_J * 100
            marker = " <-- OURS" if m == "QMIX" else ""
            _log(f"  {m:<10} {r['mean_J']:>10.2f} {r['std_J']:>8.2f} "
                 f"{r['mean_OT']:>8.1f}% {diff:>+9.1f}%{marker}")
    else:
        _log(f"{'Method':<12} {'J (mean)':>10} {'J (std)':>8} {'OnTime%':>9}")
        _log("-"*43)
        for m, r in results.items():
            _log(f"  {m:<10} {r['mean_J']:>10.2f} {r['std_J']:>8.2f} {r['mean_OT']:>8.1f}%")
        _log("  (QMIX not included in this run, so comparison column is omitted.)")
 
    os.makedirs("results", exist_ok=True)
    with open(f"results/rl_baselines_{city}.pkl", "wb") as f:
        pickle.dump(results, f)
    _log(f"Saved → results/rl_baselines_{city}.pkl")
    return results
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="bangalore")
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--eval_ep",  type=int, default=100)
    ap.add_argument("--quick_test", action="store_true", help="Run a minimal smoke test: one method, one episode, one eval episode")
    ap.add_argument("--methods", default="", help="Optional comma-separated subset of methods, e.g. DQN or DQN,QMIX")
    ap.add_argument("--log_steps", action="store_true", help="Log every step inside each episode. Off by default to keep runs readable.")
    args = ap.parse_args()
    selected_methods = [m.strip().upper() for m in args.methods.split(",") if m.strip()] or None
    evaluate_all(args.city, args.episodes, args.eval_ep, quick_test=args.quick_test, methods=selected_methods, log_steps=args.log_steps)