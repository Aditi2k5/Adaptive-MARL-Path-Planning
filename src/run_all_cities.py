import argparse, os, pickle, time
import numpy as np
from train import train
from evaluate import evaluate
 
CITIES = ["bangalore", "chennai", "hyderabad", "delhi", "mumbai"]
 
def run_all(episodes=3000, eval_ep=100):
    summary = {}
    total_start = time.time()
 
    for city in CITIES:
        print(f"\n{'='*65}")
        print(f"CITY: {city.upper()}")
        print(f"{'='*65}")
 
        # Train
        print(f"\n--- Training ({episodes} episodes) ---")
        train_logs = train(
            city_name=city,
            n_episodes=episodes,
            save_freq=max(500, episodes//6),
            log_freq=max(50, episodes//60),
        )
 
        # Evaluate
        print(f"\n--- Evaluating ({eval_ep} episodes) ---")
        model_dir = os.path.join("models", city, "final")
        results = evaluate(
            city_name=city,
            model_dir=model_dir,
            n_episodes=eval_ep,
        )
 
        summary[city] = {
            "train_final_J":       np.mean(train_logs["episode_costs"][-100:]),
            "train_final_ontime":  np.mean(train_logs["on_time_rates"][-100:]) * 100,
            "eval_results":        results,
        }
 
    # Save summary
    os.makedirs("results", exist_ok=True)
    with open("results/all_cities_summary.pkl", "wb") as f:
        pickle.dump(summary, f)
 
    # Print cross-city table
    print(f"\n\n{'='*70}")
    print("ALL CITIES — FINAL RESULTS")
    print(f"{'='*70}")
    print(f"{'City':<12} {'Train J':>9} {'Train OT%':>10} "
          f"{'Eval QMIX J':>12} {'Eval OT%':>10} {'vs Random':>10}")
    print("─" * 70)
    for city, data in summary.items():
        er = data["eval_results"]
        names = list(er.keys())
        rnd_j  = er[names[0]]["mean_cost"]
        qmx    = er[names[-1]]
        imp    = (rnd_j - qmx["mean_cost"]) / rnd_j * 100
        print(f"  {city.capitalize():<10} "
              f"{data['train_final_J']:>9.2f} "
              f"{data['train_final_ontime']:>9.1f}% "
              f"{qmx['mean_cost']:>12.2f} "
              f"{qmx['mean_on_time']:>9.1f}% "
              f"{imp:>+9.1f}%")
    print("─" * 70)
    elapsed = (time.time() - total_start) / 60
    print(f"\n  Total time: {elapsed:.1f} min")
    print("  Results saved → results/all_cities_summary.pkl")
    return summary
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes",      type=int, default=3000)
    ap.add_argument("--eval_episodes", type=int, default=100)
    args = ap.parse_args()
    run_all(args.episodes, args.eval_episodes)