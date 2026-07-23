import sys
# Configure standard streams to handle Unicode characters gracefully on all platforms
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            try:
                stream.reconfigure(errors='replace')
            except Exception:
                pass

# ── Location ──────────────────────────────────────────────────────────────────
DARK_STORE_LOCATION = (12.9716, 77.5946)   # Koramangala, Bangalore
CITY                = "Bangalore, India"
SERVICE_RADIUS_KM   = 3.0   # All customers within 3 km (real constraint)

# ── Rider Parameters ──────────────────────────────────────────────────────────
N_RIDERS   = 5
K_CAPACITY = 1      # One order per rider at a time
V_SPEED    = 15.0   # km/hr urban average (realistic)

# ── Route Alternatives ────────────────────────────────────────────────────────
# Each delivery has K_ROUTES alternative road paths pre-computed from OSM.
# The RL agent selects WHICH route to take based on current conditions.
K_ROUTES = 3   # Number of alternative routes per delivery

# Route type properties (used in environment)
ROUTE_TYPES = {
    'arterial':    {'base_traffic_mult': 1.40, 'base_weather_mult': 1.05,
                    'flood_risk': 0.0,  'description': 'Main road, fast normally but congested'},
    'residential': {'base_traffic_mult': 1.10, 'base_weather_mult': 1.15,
                    'flood_risk': 0.0,  'description': 'Back roads, consistent but longer'},
    'shortcut':    {'base_traffic_mult': 1.20, 'base_weather_mult': 1.60,
                    'flood_risk': 0.65, 'description': 'Underpass/shortcut, fast but floods'},
}

# ── Deadline Parameters ───────────────────────────────────────────────────────
# 20-minute SLA is Blinkit/Zepto standard
# Single delivery (3km max, 15km/hr, traffic 1.3×): ≈ 16 min
DEADLINE_DURATIONS = [15, 20, 30]    # minutes
DEADLINE_WEIGHTS   = [0.20, 0.55, 0.25]

# ── Cost Parameters ───────────────────────────────────────────────────────────
C_KM   = 0.20    # ₹/km
P_LATE = 5.0    # ₹/minute late

# ── θ Parameters (from paper formulation) ────────────────────────────────────
THETA_1 = 5.0   # urgency weight           (Ulmer CFA)
THETA_4 = 8.0   # weather risk penalty     (YOUR contribution)
THETA_5 = 6.0   # breakdown risk penalty   (YOUR contribution)

# ── Traffic Patterns ──────────────────────────────────────────────────────────
TRAFFIC_MULTIPLIERS = {
    'free':     1.0,
    'moderate': 1.15,
    'peak':     1.30,
}
TRAFFIC_SCHEDULE = {
    ( 0,  7): 'free',
    ( 7, 10): 'peak',
    (10, 17): 'moderate',
    (17, 21): 'peak',
    (21, 24): 'moderate',
}

# ── Weather States ────────────────────────────────────────────────────────────
WEATHER_STATES = {
    0: {'name': 'clear',      'base_mult': 1.00, 'risk': 0},
    1: {'name': 'drizzle',    'base_mult': 1.15, 'risk': 1},
    2: {'name': 'heavy_rain', 'base_mult': 1.40, 'risk': 3},
}
WEATHER_PROBS      = [0.65, 0.25, 0.10]
WEATHER_TRANSITION = {   # Markov chain: weather evolves across timesteps
    0: [0.90, 0.08, 0.02],
    1: [0.15, 0.75, 0.10],
    2: [0.05, 0.20, 0.75],
}

# ── RL / MARL Parameters ─────────────────────────────────────────────────────
GAMMA         = 0.95
LEARNING_RATE = 3e-4
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = 0.999    # reaches 0.05 around episode 900
REWARD_SCALE  = 50.0
# State / action dimensions
# State per agent: rider features + delivery context + route options
STATE_DIM  = 25    # see agent.py for exact layout
ACTION_DIM = K_ROUTES   # 3 route choices

HIDDEN_DIM = 128

# Training
BATCH_SIZE         = 64
REPLAY_BUFFER_SIZE = 20_000
TARGET_UPDATE_FREQ = 200

# QMIX
QMIX_EMBED_DIM   = 32
GLOBAL_STATE_DIM = 28

# Episode
EPISODE_LENGTH    = 480   # 8-hour shift (minutes)
ORDERS_PER_SHIFT  = 50    # total orders simulated per shift per rider
LAMBDA_ARRIVAL    = 1.5   # orders/minute (Poisson)
TIMESTEP_DURATION = 1     # minutes

# Paths
ROAD_NETWORK_CACHE = 'data/road_network.pkl'
MODELS_DIR         = 'models/'
LOGS_DIR           = 'logs/'
RESULTS_DIR        = 'results/'
RANDOM_SEED        = 42
ACTIVE_CITY        = "bangalore"

def load_city(city_name=None):
        """Load city-specific parameters into globals."""
        global DARK_STORE_LOCATION, CITY, SERVICE_RADIUS_KM, V_SPEED
        global WEATHER_PROBS, WEATHER_MULTS, ROUTE_FLOOD_RISKS
        global ROUTE_DIST_FACTORS, ROUTE_TRAFFIC_MULTS, ROUTE_WEATHER_MULTS
        global SHORTCUT_FLOOD_TRIGGER, SHORTCUT_FLOOD_MULT
        global ROAD_FACTOR, ROAD_NETWORK_CACHE, ACTIVE_CITY

        from multi_city_configs import get_city
        name = (city_name or ACTIVE_CITY).lower()
        cfg  = get_city(name)
        ACTIVE_CITY            = name
        DARK_STORE_LOCATION    = cfg["dark_store"]
        CITY                   = cfg["osm_query"]
        SERVICE_RADIUS_KM      = cfg["service_radius"]
        V_SPEED                = cfg["v_speed"]
        WEATHER_PROBS          = cfg["weather_probs"]
        WEATHER_MULTS          = cfg["weather_mults"]
        ROUTE_FLOOD_RISKS      = cfg["route_flood_risks"]
        ROUTE_DIST_FACTORS     = cfg["route_dist_factors"]
        ROUTE_TRAFFIC_MULTS    = cfg["route_traffic_mults"]
        ROUTE_WEATHER_MULTS    = cfg["route_weather_mults"]
        SHORTCUT_FLOOD_TRIGGER = cfg["shortcut_flood_trigger"]
        SHORTCUT_FLOOD_MULT    = cfg["shortcut_flood_mult"]
        ROAD_FACTOR            = cfg["road_factor"]
        ROAD_NETWORK_CACHE     = f"data/road_network_{name}.pkl"

# Initialize with default city parameters
load_city(ACTIVE_CITY)