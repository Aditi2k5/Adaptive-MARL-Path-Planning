import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
CITIES = ["bangalore", "chennai", "hyderabad", "delhi", "mumbai"]
CITY_LABELS = ["Bangalore", "Chennai", "Hyderabad", "Delhi", "Mumbai"]
COLORS = ["#2E75B6", "#ED7D31", "#70AD47", "#FFC000", "#FF0000"]
 
os.makedirs("results", exist_ok=True)
 
def load_results():
    path = "results/all_cities_summary.pkl"
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.")
        print("Run:  python run_all_cities.py  first.")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
 
def plot_cost_comparison(summary):
    """Bar chart: QMIX J vs Random J per city."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(CITIES))
    w = 0.35
    random_Js = []
    qmix_Js   = []
    for city in CITIES:
        er = summary[city]["eval_results"]
        names = list(er.keys())
        random_Js.append(er[names[0]]["mean_cost"])
        qmix_Js.append(er[names[-1]]["mean_cost"])
    bars1 = ax.bar(x - w/2, random_Js, w, label="Random Baseline",
                   color="#ADB9CA", edgecolor="white")
    bars2 = ax.bar(x + w/2, qmix_Js,   w, label="QMIX MARL (Ours)",
                   color=COLORS, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(CITY_LABELS, fontsize=12)
    ax.set_ylabel("Cost J (₹)", fontsize=12)
    ax.set_title("Objective Function J: QMIX vs Random — All Cities", fontsize=14)
    ax.legend(fontsize=11); ax.grid(axis="y", alpha=0.3)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"₹{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig("results/compare_cost_all_cities.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: results/compare_cost_all_cities.png")
 
def plot_ontime_comparison(summary):
    """Bar chart: on-time rate per city per strategy."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(CITIES)); w = 0.2
    strategies = ["Random Route", "Static Dijkstra",
                  "Weather-Aware Heuristic", "QMIX MARL (Trained)"]
    strat_colors = ["#ADB9CA", "#5B9BD5", "#ED7D31", "#70AD47"]
    for si, (sname, col) in enumerate(zip(strategies, strat_colors)):
        ots = []
        for city in CITIES:
            er = summary[city]["eval_results"]
            for k, v in er.items():
                if sname.split(" ")[0].lower() in k.lower() or \
                   (si == 3 and "qmix" in k.lower()):
                    ots.append(v["mean_on_time"])
                    break
            else:
                ots.append(0)
        offset = (si - 1.5) * w
        ax.bar(x + offset, ots, w, label=sname, color=col, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(CITY_LABELS, fontsize=12)
    ax.set_ylabel("On-time Delivery Rate (%)", fontsize=12)
    ax.set_title("On-time Rate by Strategy — All Cities", fontsize=14)
    ax.axhline(85, color="red", ls="--", lw=1, label="85% target")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig("results/compare_ontime_all_cities.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: results/compare_ontime_all_cities.png")
 
def plot_route_usage_rain(summary):
    """Stacked bar: route usage in heavy rain per city (QMIX only)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(CITIES)); w = 0.5
    r0s=[]; r1s=[]; r2s=[]
    for city in CITIES:
        er = summary[city]["eval_results"]
        for k, v in er.items():
            if "qmix" in k.lower():
                rw = v["route_by_weather"].get(2, {})
                tot = sum(rw.values()) + 1e-9
                r0s.append(rw.get(0,0)/tot*100)
                r1s.append(rw.get(1,0)/tot*100)
                r2s.append(rw.get(2,0)/tot*100)
                break
        else:
            r0s.append(33); r1s.append(33); r2s.append(34)
    ax.bar(x, r0s, w, label="R0 Arterial",     color="#2E75B6")
    ax.bar(x, r1s, w, label="R1 Residential",  color="#70AD47", bottom=r0s)
    ax.bar(x, r2s, w, label="R2 Shortcut",     color="#FF0000",
           bottom=[a+b for a,b in zip(r0s,r1s)])
    ax.set_xticks(x); ax.set_xticklabels(CITY_LABELS, fontsize=12)
    ax.set_ylabel("Route Usage (%)", fontsize=12)
    ax.set_title("QMIX Route Usage in Heavy Rain — All Cities\n"
                 "(R2/Shortcut should be near 0% — learned flood avoidance)",
                 fontsize=13)
    ax.legend(fontsize=11); ax.set_ylim(0, 105)
    ax.axhline(10, color="red", ls="--", lw=1.2, label="10% shortcut threshold")
    plt.tight_layout()
    plt.savefig("results/compare_route_rain_all_cities.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: results/compare_route_rain_all_cities.png")
 
def plot_improvement_over_random(summary):
    """Horizontal bar: % improvement of QMIX over random per city."""
    fig, ax = plt.subplots(figsize=(10, 6))
    improvements = []
    for city in CITIES:
        er = summary[city]["eval_results"]
        names = list(er.keys())
        rnd_j = er[names[0]]["mean_cost"]
        qmx_j = er[names[-1]]["mean_cost"]
        improvements.append((rnd_j - qmx_j) / rnd_j * 100)
    y = np.arange(len(CITIES))
    bars = ax.barh(y, improvements, color=COLORS, edgecolor="white")
    ax.set_yticks(y); ax.set_yticklabels(CITY_LABELS, fontsize=12)
    ax.set_xlabel("Cost J Improvement vs Random (%)", fontsize=12)
    ax.set_title("QMIX Improvement over Random Baseline — All Cities", fontsize=14)
    ax.axvline(0, color="black", lw=0.8)
    ax.grid(axis="x", alpha=0.3)
    for bar, imp in zip(bars, improvements):
        ax.text(imp + 0.5, bar.get_y() + bar.get_height()/2,
                f"{imp:+.1f}%", va="center", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/compare_improvement_all_cities.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: results/compare_improvement_all_cities.png")
 
def print_summary_table(summary):
    print(f"\n{'='*80}")
    print("CROSS-CITY SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"  {'City':<12} {'Random J':>9} {'QMIX J':>8} {'Improvement':>12} "
          f"{'OnTime QMIX':>12} {'Road Factor':>12}")
    print("  " + "─"*72)
    from multi_city_configs import CITIES as CITY_CFG
    for city in CITIES:
        er = summary[city]["eval_results"]
        names = list(er.keys())
        rnd   = er[names[0]]
        qmx   = er[names[-1]]
        imp   = (rnd["mean_cost"] - qmx["mean_cost"]) / rnd["mean_cost"] * 100
        rf    = CITY_CFG[city]["road_factor"]
        print(f"  {city.capitalize():<12} "
              f"{rnd['mean_cost']:>8.2f}  "
              f"{qmx['mean_cost']:>7.2f}  "
              f"{imp:>+10.1f}%  "
              f"{qmx['mean_on_time']:>10.1f}%  "
              f"{rf:>12.2f}")
    print(f"{'='*80}")
 
if __name__ == "__main__":
    print("Generating cross-city comparison plots...\n")
    summary = load_results()
    if summary is None:
        exit(1)
    print_summary_table(summary)
    print("\nGenerating plots:")
    plot_cost_comparison(summary)
    plot_ontime_comparison(summary)
    plot_route_usage_rain(summary)
    plot_improvement_over_random(summary)
    print("\n✓ All comparison plots saved to results/")
    print("\nFiles generated:")
    print("  results/compare_cost_all_cities.png")
    print("  results/compare_ontime_all_cities.png")
    print("  results/compare_route_rain_all_cities.png")
    print("  results/compare_improvement_all_cities.png")