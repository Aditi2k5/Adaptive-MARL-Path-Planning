## Adaptive MARL Path Planning — QMIX Multi-Agent Coordination
This repository contains a PyTorch implementation of a **QMIX-based Multi-Agent Reinforcement Learning (MARL)** system designed for dynamic path planning, agent coordination, and collision avoidance in non-stationary spatial environments. 

> ### 🎯 Context & Research Scope
> 
> My research explores autonomous spatial systems across two distinct layers:
> * **Perception Layer:** Covered in my *Transient Object Detection* work (3D scene reconstruction).
> * **Decision-Making Layer (This Repository):** Focuses on multi-agent spatial coordination, path planning, and collision avoidance in dynamic environments using QMIX.



**Contents**
- `train.py` — training loop that saves models and `results/training_log.pkl`.
- `evaluate.py` — runs evaluation episodes and writes `results/eval_results.pkl`.
- `visualize.py` — generates publication-quality figures from saved results.
- `road_network.py` — OSM-backed or stubbed road network utilities.
- `config.py` — experiment configuration and paths.
- `models/`, `results/`, `logs/` — model checkpoints, plots, and logs.

**Prerequisites**
- Python 3.11+ (virtualenv recommended)
- Create and activate a venv in the repo root:

```bash
python -m venv .venv
.
# On Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

- Install dependencies:

```bash
pip install -r requirements.txt
```
### Option A: Local Virtual Environment 

Python 3.11+ 
Create and activate a venv in the repo root:
python -m venv .venv
.
# On Windows PowerShell
.\.venv\Scripts\Activate.ps1
Install dependencies:
pip install -r requirements.txt

### Option B: Docker Setup 


Launch the container terminal 

docker compose run --rm app bash 


**Quick Usage**

- Train (example):

```bash
python train.py --episodes 3000
```

- Evaluate (example):

```bash
python evaluate.py --model_dir models/final --n_episodes 100
```

- Visualize saved results (generates PNGs in `results/`):

```bash
python visualize.py
```

- Plot demo routes on a map (requires `osmnx` and `networkx` or falls back to a stub):

```bash
python visualize.py --plot_routes_on_map
```

You can also provide a pickle file containing `routes_data` (list of dicts with `waypoints`, `color`, `label`) using `--routes routes.pkl`.

### 📊 Experimental Analysis & Baselines (`/scripts/`)

* `run_all_cities.py`: **Multi-City Pipeline** — Sequentially trains and evaluates the model across all target road networks to verify cross-city generalizability.
* `run_rl_baselines.py`: **Algorithm Benchmarking** — Runs comparative evaluations against standard MARL architectures (DQN, IQL, VDN, MAPPO) under identical environment constraints.
* `ablation_study.py`: **Feature Validation** — Systematically toggles individual dynamic inputs (e.g., weather signals, flood risk levels) to isolate and prove the value of each component.
* `compare_cities.py`: **Metric Aggregation** — Compiles raw evaluation data into cross-city comparison tables and generates visual performance plots (`.png`).


**Output Files**
- `results/learning_curves.png` — three-panel plot: Cost J, On-time rate (with 90% target line), and TD loss. Shows raw values and a moving average.
- `results/epsilon_decay.png` — epsilon values over episodes (exploration → exploitation).
- `results/comparison_chart.png` — bar chart comparing strategies (mean cost and on-time rate).
- `results/training_log.pkl` — pickled training metrics used by `visualize.py`.
- `results/eval_results.pkl` — pickled evaluation summary used by `visualize.py`.

