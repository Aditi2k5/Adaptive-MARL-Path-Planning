import argparse, logging, os, pickle
import numpy as np
import torch
from tqdm import tqdm
import config
from qmix import QMIXCoordinator
from environment import QuickCommerceEnv
from road_network import RoadNetwork
from agent import build_agent_state, build_global_state
 
ABLATIONS = {
    "Full QMIX":           {"theta4": True,  "theta5": True,  "flood_replay": True,  "qmix_mix": True},
    "No θ4 (no weather)":  {"theta4": False, "theta5": True,  "flood_replay": True,  "qmix_mix": True},
    "No θ5 (no breakdown)":{"theta4": True,  "theta5": False, "flood_replay": True,  "qmix_mix": True},
    "No flood replay":     {"theta4": True,  "theta5": True,  "flood_replay": False,  "qmix_mix": True},
    "IQL (no mixing)":     {"theta4": True,  "theta5": True,  "flood_replay": True,  "qmix_mix": False},
}

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required. This ablation script is configured to run on GPU only.")

CUDA_DEVICE = torch.device("cuda")

LOGGER = logging.getLogger("ablation_runner")


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
    file_handler = logging.FileHandler(os.path.join("results", f"ablation_{city}.log"), mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(console_handler)
    LOGGER.addHandler(file_handler)
    return LOGGER


def _log(message):
    LOGGER.info(message)


def _device_banner(device: torch.device) -> str:
    idx = torch.cuda.current_device()
    return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
 
def compute_J_ablated(env, theta4_on, theta5_on):
    """Compute J with specific terms enabled/disabled."""
    J = 0.0
    for r in env.riders:
        J += config.C_KM * r.total_distance
        if theta5_on:
            J += config.THETA_5 * (1 - r.health) * r.total_distance
    for o in env.orders.values():
        if o.delivered and o.is_late():
            J += config.P_LATE * o.delay()
        if o.delivered and o.chosen_route is not None and theta4_on:
            J += o.routes[o.chosen_route].weather_risk_penalty(env.weather)
    for o in env.pending_orders():
        J += config.THETA_1 * o.urgency(env.t)
    return max(0.0, J)
 
 
def run_ablation(city, episodes=3000, eval_ep=100):
    _configure_logger(city)
    _log(f"Using CUDA device: {_device_banner(CUDA_DEVICE)}")
    _log(f"Preparing {city} city config and road network...")
    config.load_city(city)
    network = RoadNetwork(use_cache=True)
    env     = QuickCommerceEnv(network=network)
 
    results = {}
 
    for ablation_name, flags in ABLATIONS.items():
        _log(f"\nTraining: {ablation_name}")
 
        # Modify environment reward for this ablation
        coord = QMIXCoordinator(n_agents=config.N_RIDERS, device=CUDA_DEVICE)
        eps   = config.EPSILON_START
        AJ=[]; AOT=[]
 
        with tqdm(total=episodes, desc=f"{ablation_name} train", unit="ep", leave=False) as bar:
            for ep in range(1, episodes + 1):
                env.reset()
                while env.t < config.EPISODE_LENGTH:
                    idle    = env.idle_riders()
                    pending = env.pending_orders()
                    if not idle or not pending:
                        env.step({}); continue

                    pending_sorted = sorted(pending, key=lambda o: o.urgency(env.t), reverse=True)
                    pairs = list(zip(idle, pending_sorted))
                    rider_order_map = {r.id: o for r,o in pairs}

                    obs_list = []
                    for rider in env.riders:
                        order = rider_order_map.get(rider.id)
                        if order is not None and rider.is_idle:
                            obs_list.append(build_agent_state(rider, order, env.traffic, env.weather, env.t, network))
                        else:
                            obs_list.append(np.zeros(config.STATE_DIM, dtype=np.float32))

                    actions = coord.select_actions(obs_list, epsilon=eps)
                    path_decisions = {r.id: actions[r.id] for r,o in pairs}
                    gs_before = build_global_state(env.riders, pending, env.traffic, env.weather, env.t)
                    _, reward, done, _ = env.step(path_decisions)
                    gs_after  = build_global_state(env.riders, env.pending_orders(), env.traffic, env.weather, env.t)
                    nobs_list = []
                    for rider in env.riders:
                        nobs_list.append(np.zeros(config.STATE_DIM, dtype=np.float32))

                    # Override reward with ablated J
                    abl_J  = compute_J_ablated(env, flags["theta4"], flags["theta5"])
                    reward = -abl_J / config.REWARD_SCALE

                    coord.push_transition(
                        gs_before, np.stack(obs_list), np.array(actions),
                        reward, gs_after, np.stack(nobs_list), bool(done),
                        weather_state=env.weather if flags["flood_replay"] else 0
                    )
                    coord.train()
                    if done: break

                J = compute_J_ablated(env, flags["theta4"], flags["theta5"])
                delivered = [o for o in env.orders.values() if o.delivered]
                ot = sum(1 for o in delivered if not o.is_late()) / max(1, len(delivered))
                AJ.append(J); AOT.append(ot)
                eps = max(config.EPSILON_END, eps * config.EPSILON_DECAY)
                if ep % config.TARGET_UPDATE_FREQ == 0:
                    coord.sync_targets()
                if ep % 25 == 0 or ep == 1 or ep == episodes:
                    _log(f"[{ablation_name} | train ep{ep}/{episodes}] J={np.mean(AJ[-100:]):.1f} OT={np.mean(AOT[-100:])*100:.1f}% eps={eps:.4f}")

                bar.set_postfix(J=f"{J:.1f}", OT=f"{ot*100:.1f}%", eps=f"{eps:.4f}")
                bar.update(1)
 
        # Evaluate
        Js=[]; OTs=[]
        with tqdm(total=eval_ep, desc=f"{ablation_name} eval", unit="ep", leave=False) as bar:
            for eval_idx in range(1, eval_ep + 1):
                env.reset()
                while env.t < config.EPISODE_LENGTH:
                    idle = env.idle_riders(); pending = env.pending_orders()
                    if not idle or not pending: env.step({}); continue
                    pending_sorted = sorted(pending, key=lambda o: o.urgency(env.t), reverse=True)
                    pairs = list(zip(idle, pending_sorted))
                    rider_order_map = {r.id: o for r,o in pairs}
                    obs_list = []
                    for rider in env.riders:
                        order = rider_order_map.get(rider.id)
                        if order is not None and rider.is_idle:
                            obs_list.append(build_agent_state(rider, order, env.traffic, env.weather, env.t, network))
                        else:
                            obs_list.append(np.zeros(config.STATE_DIM, dtype=np.float32))
                    actions = coord.select_actions(obs_list, epsilon=0.0)
                    path_decisions = {r.id: actions[r.id] for r,o in pairs}
                    _, _, done, _ = env.step(path_decisions)
                    if done: break
                J  = compute_J_ablated(env, flags["theta4"], flags["theta5"])
                delivered = [o for o in env.orders.values() if o.delivered]
                ot = sum(1 for o in delivered if not o.is_late()) / max(1, len(delivered))
                Js.append(J); OTs.append(ot)
                if eval_idx % 10 == 0 or eval_idx == 1 or eval_idx == eval_ep:
                    _log(f"[{ablation_name} | eval ep{eval_idx}/{eval_ep}] J={J:.1f} OT={ot*100:.1f}%")
                bar.set_postfix(J=f"{J:.1f}", OT=f"{ot*100:.1f}%")
                bar.update(1)
 
        results[ablation_name] = {
            "mean_J": np.mean(Js), "std_J": np.std(Js),
            "mean_OT": np.mean(OTs)*100, "all_J": Js
        }
 
    # Print ablation table
    _log(f"\n{'='*65}")
    _log(f"ABLATION STUDY — {city.upper()}")
    _log(f"{'='*65}")
    base = results["Full QMIX"]["mean_J"]
    _log(f"{'Configuration':<30} {'J (₹)':>9} {'Std':>7} {'OnTime':>8} {'Δ vs Full':>10}")
    _log("-"*65)
    for name, r in results.items():
        delta = (r["mean_J"] - base) / base * 100
        marker = " ← FULL" if name == "Full QMIX" else ""
        _log(f"  {name:<28} {r['mean_J']:>8.2f} {r['std_J']:>7.2f} "
             f"{r['mean_OT']:>7.1f}% {delta:>+9.1f}%{marker}")
 
    os.makedirs("results", exist_ok=True)
    with open(f"results/ablation_{city}.pkl", "wb") as f:
        pickle.dump(results, f)
    _log(f"\n  Saved → results/ablation_{city}.pkl")
    return results
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city",     default="bangalore")
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--eval_ep",  type=int, default=100)
    args = ap.parse_args()
    run_ablation(args.city, args.episodes, args.eval_ep)
 