from __future__ import annotations
import argparse
import os
import pickle
import time
from typing import Dict, List
import numpy as np
import torch
from tqdm import tqdm
import config
from agent import build_agent_state, build_global_state
from data_structures import make_fleet, Order
from environment import QuickCommerceEnv
from qmix import QMIXCoordinator
from road_network import RoadNetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def collect_observations(env: QuickCommerceEnv,
                         rider_order_map: Dict[int, Order],
                         network: RoadNetwork) -> List[np.ndarray]:
    obs_list = []
    for rider in env.riders:
        order = rider_order_map.get(rider.id)
        if order is not None:
            obs = build_agent_state(
                rider, order,
                env.traffic, env.weather, env.t, network
            )
        else:
            obs = np.zeros(config.STATE_DIM, dtype=np.float32)
        obs_list.append(obs)
    return obs_list


def train(n_episodes:  int = 3000,
          save_freq:   int = 500,
          log_freq:    int = 50) -> Dict:
    """
    Train QMIX agents for adaptive path planning.
    """
    os.makedirs(config.MODELS_DIR,  exist_ok=True)
    os.makedirs(config.LOGS_DIR,    exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("=" * 65)
    print("QMIX TRAINING — Adaptive Path Planning for Quick Commerce")
    print("=" * 65)
    print(f"  City          : {config.CITY}")
    print(f"  Service radius: {config.SERVICE_RADIUS_KM} km")
    print(f"  Riders        : {config.N_RIDERS}")
    print(f"  Route choices : {config.K_ROUTES}  (arterial / residential / shortcut)")
    print(f"  Episodes      : {n_episodes}")
    print(f"  Deadline SLAs : {config.DEADLINE_DURATIONS} minutes")
    print("=" * 65 + "\n")

    network     = RoadNetwork(use_cache=True)
    env         = QuickCommerceEnv(network=network)
    coordinator = QMIXCoordinator(n_agents=config.N_RIDERS, device=DEVICE)

    logs = {
        "episode_costs":     [],
        "on_time_rates":     [],
        "distances":         [],
        "losses":            [],
        "epsilons":          [],
        "route_usage":       [],   # [Counter per episode]
    }

    start_time = time.time()

    for ep in tqdm(range(1, n_episodes + 1), desc="Training"):

        env.reset()
        ep_loss_sum   = 0.0
        ep_loss_count = 0
        from collections import Counter
        ep_route_usage = Counter()

        # Run timesteps until shift ends
        while env.t < config.EPISODE_LENGTH:

            idle_riders  = env.idle_riders()
            pending      = env.pending_orders()

            if not idle_riders or not pending:
                # No decisions to make — advance time
                env.step({})
                continue
            pending_sorted = sorted(
                pending,
                key=lambda o: o.urgency(env.t),
                reverse=True
            )
            pairs = list(zip(idle_riders, pending_sorted))

            rider_order_map = {r.id: o for r, o in pairs}

            obs_list   = collect_observations(env, rider_order_map, network)
            gs_before  = build_global_state(
                env.riders, pending, env.traffic, env.weather, env.t
            )

            actions = coordinator.select_actions(obs_list)

            # Build dispatch: rider_id → route_index
            dispatch = {}
            for rider, order in pairs:
                route_idx = actions[rider.id]
                dispatch[rider.id] = (order.id, route_idx)
                ep_route_usage[route_idx] += 1

            path_decisions = {}
            order_assignments = {}
            for rider_id, (order_id, route_idx) in dispatch.items():
                order_assignments[rider_id] = order_id
                path_decisions[rider_id]    = route_idx
            _, reward, done, info = env.step(path_decisions)

            gs_after   = build_global_state(
                env.riders, env.pending_orders(),
                env.traffic, env.weather, env.t
            )
            next_obs   = collect_observations(env, {}, network)

            obs_arr    = np.stack(obs_list)               # (N, STATE_DIM)
            acts_arr   = np.array(actions)                # (N,)
            next_arr   = np.stack(next_obs)               # (N, STATE_DIM)

            coordinator.push_transition(
                global_state  = gs_before,
                obs           = obs_arr,
                actions       = acts_arr,
                team_reward   = float(reward),
                next_global   = gs_after,
                next_obs      = next_arr,
                done          = bool(done),
            )

            loss = coordinator.train()
            if loss is not None:
                ep_loss_sum   += loss
                ep_loss_count += 1

            if done:
                break

        ep_cost    = env.compute_J()
        ep_on_time = env._on_time_rate()
        ep_dist    = sum(r.total_distance for r in env.riders)
        ep_loss    = ep_loss_sum / ep_loss_count if ep_loss_count else 0.0

        logs["episode_costs"].append(ep_cost)
        logs["on_time_rates"].append(ep_on_time)
        logs["distances"].append(ep_dist)
        logs["losses"].append(ep_loss)
        logs["epsilons"].append(coordinator.agents[0].epsilon)
        logs["route_usage"].append(dict(ep_route_usage))

        coordinator.decay_epsilon()

        # Sync target networks periodically
        if ep % config.TARGET_UPDATE_FREQ == 0:
            coordinator.sync_targets()

        # Progress logging
        if ep % log_freq == 0:
            w = min(log_freq, ep)
            avg_cost    = np.mean(logs["episode_costs"][-w:])
            avg_on_time = np.mean(logs["on_time_rates"][-w:]) * 100
            avg_dist    = np.mean(logs["distances"][-w:])
            avg_loss    = np.mean(logs["losses"][-w:])
            eps         = coordinator.agents[0].epsilon

            tqdm.write(
                f"Ep {ep:5d} | "
                f"J=₹{avg_cost:7.2f} | "
                f"OnTime={avg_on_time:5.1f}% | "
                f"Dist={avg_dist:5.2f}km | "
                f"Loss={avg_loss:.4f} | "
                f"ε={eps:.3f}"
            )

        # Save checkpoint
        if ep % save_freq == 0:
            ckpt_dir = os.path.join(config.MODELS_DIR, f"ep{ep}")
            coordinator.save(ckpt_dir)
            log_path = os.path.join(config.LOGS_DIR, f"log_ep{ep}.pkl")
            with open(log_path, "wb") as f:
                pickle.dump(logs, f)

    elapsed = time.time() - start_time
    w       = min(100, n_episodes)

    print("\n" + "=" * 65)
    print("TRAINING COMPLETE")
    print("=" * 65)
    print(f"  Total time   : {elapsed/60:.1f} min")
    print(f"  Final Cost J : ₹{np.mean(logs['episode_costs'][-w:]):.2f}  "
          f"(±{np.std(logs['episode_costs'][-w:]):.2f})")
    print(f"  On-time rate : {np.mean(logs['on_time_rates'][-w:])*100:.1f}%")
    print(f"  Avg distance : {np.mean(logs['distances'][-w:]):.2f} km/episode")

    # Route usage in final episodes
    from collections import Counter
    final_routes = Counter()
    for ru in logs["route_usage"][-w:]:
        for k, v in ru.items():
            final_routes[k] += v
    total_r = sum(final_routes.values()) + 1
    print(f"  Route usage  : "
          f"R0(arterial)={final_routes.get(0,0)/total_r*100:.0f}%  "
          f"R1(residential)={final_routes.get(1,0)/total_r*100:.0f}%  "
          f"R2(shortcut)={final_routes.get(2,0)/total_r*100:.0f}%")
    print("=" * 65)

    # Save final model and full log
    coordinator.save(os.path.join(config.MODELS_DIR, "final"))
    with open(os.path.join(config.RESULTS_DIR, "training_log.pkl"), "wb") as f:
        pickle.dump(logs, f)
    print(f"\n✓ Model saved  → {config.MODELS_DIR}final/")
    print(f"✓ Logs saved   → {config.RESULTS_DIR}training_log.pkl")

    return logs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train QMIX agents for adaptive path planning"
    )
    parser.add_argument("--episodes",  type=int, default=3000,
                        help="Number of training episodes (default: 3000)")
    parser.add_argument("--save_freq", type=int, default=500,
                        help="Save checkpoint every N episodes (default: 500)")
    parser.add_argument("--log_freq",  type=int, default=50,
                        help="Print progress every N episodes (default: 50)")
    args = parser.parse_args()

    train(
        n_episodes = args.episodes,
        save_freq  = args.save_freq,
        log_freq   = args.log_freq,
    )