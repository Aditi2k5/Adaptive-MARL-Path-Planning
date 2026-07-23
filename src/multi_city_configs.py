"""
Per-city parameters for multi-city adaptive path planning.
Covers: Bangalore, Chennai, Hyderabad, Delhi, Mumbai.
"""

CITIES = {
    "bangalore": {
        "name": "Bangalore",
        "osm_query": "Bangalore, India",
        "dark_store": (12.9716, 77.5946),
        "service_radius": 3.0,
        "road_factor": 1.35,
        "v_speed": 15.0,
        "traffic_peak_mult": 1.30,
        "weather_probs": [0.65, 0.25, 0.10],
        "weather_mults": {0: 1.00, 1: 1.15, 2: 1.40},
        "route_flood_risks": {"arterial": 0.00, "residential": 0.00, "shortcut": 0.65},
        "route_dist_factors": {"arterial": 1.30, "residential": 1.38, "shortcut": 1.18},
        "route_traffic_mults": {"arterial": 1.40, "residential": 1.10, "shortcut": 1.20},
        "route_weather_mults": {"arterial": 1.05, "residential": 1.15, "shortcut": 1.60},
        "shortcut_flood_trigger": 1,
        "shortcut_flood_mult": 1.50,
        "notes": "IT corridor peak. Underpasses flood Jun-Sep monsoon.",
    },
    "chennai": {
        "name": "Chennai",
        "osm_query": "Chennai, India",
        "dark_store": (13.0827, 80.2707),
        "service_radius": 3.0,
        "road_factor": 1.28,
        "v_speed": 15.0,
        "traffic_peak_mult": 1.25,
        "weather_probs": [0.60, 0.25, 0.15],
        "weather_mults": {0: 1.00, 1: 1.20, 2: 1.50},
        "route_flood_risks": {"arterial": 0.10, "residential": 0.05, "shortcut": 0.70},
        "route_dist_factors": {"arterial": 1.28, "residential": 1.35, "shortcut": 1.15},
        "route_traffic_mults": {"arterial": 1.35, "residential": 1.10, "shortcut": 1.15},
        "route_weather_mults": {"arterial": 1.10, "residential": 1.20, "shortcut": 1.70},
        "shortcut_flood_trigger": 1,
        "shortcut_flood_mult": 1.50,
        "notes": "Cyclone rain Oct-Dec. Coastal flooding. Higher flood severity.",
    },
    "hyderabad": {
        "name": "Hyderabad",
        "osm_query": "Hyderabad, India",
        "dark_store": (17.3850, 78.4867),
        "service_radius": 3.0,
        "road_factor": 1.30,
        "v_speed": 16.0,
        "traffic_peak_mult": 1.30,
        "weather_probs": [0.70, 0.22, 0.08],
        "weather_mults": {0: 1.00, 1: 1.12, 2: 1.35},
        "route_flood_risks": {"arterial": 0.00, "residential": 0.00, "shortcut": 0.45},
        "route_dist_factors": {"arterial": 1.25, "residential": 1.32, "shortcut": 1.15},
        "route_traffic_mults": {"arterial": 1.30, "residential": 1.08, "shortcut": 1.18},
        "route_weather_mults": {"arterial": 1.05, "residential": 1.12, "shortcut": 1.45},
        "shortcut_flood_trigger": 2,
        "shortcut_flood_mult": 2.00,
        "notes": "Moderate monsoon. HITEC City congestion. Least flood risk.",
    },
    "delhi": {
        "name": "Delhi",
        "osm_query": "New Delhi, India",
        "dark_store": (28.6139, 77.2090),
        "service_radius": 3.0,
        "road_factor": 1.40,
        "v_speed": 13.0,
        "traffic_peak_mult": 1.40,
        "weather_probs": [0.60, 0.28, 0.12],
        "weather_mults": {0: 1.00, 1: 1.20, 2: 1.55},
        "route_flood_risks": {"arterial": 0.05, "residential": 0.05, "shortcut": 0.40},
        "route_dist_factors": {"arterial": 1.35, "residential": 1.42, "shortcut": 1.20},
        "route_traffic_mults": {"arterial": 1.50, "residential": 1.15, "shortcut": 1.25},
        "route_weather_mults": {"arterial": 1.10, "residential": 1.15, "shortcut": 1.50},
        "shortcut_flood_trigger": 2,
        "shortcut_flood_mult": 1.80,
        "notes": "Worst traffic in India. Winter fog AND monsoon rain disruptions.",
    },
    "mumbai": {
        "name": "Mumbai",
        "osm_query": "Mumbai, India",
        "dark_store": (19.0760, 72.8777),
        "service_radius": 2.5,
        "road_factor": 1.45,
        "v_speed": 12.0,
        "traffic_peak_mult": 1.45,
        "weather_probs": [0.55, 0.28, 0.17],
        "weather_mults": {0: 1.00, 1: 1.25, 2: 1.65},
        "route_flood_risks": {"arterial": 0.20, "residential": 0.15, "shortcut": 0.85},
        "route_dist_factors": {"arterial": 1.40, "residential": 1.48, "shortcut": 1.22},
        "route_traffic_mults": {"arterial": 1.50, "residential": 1.20, "shortcut": 1.30},
        "route_weather_mults": {"arterial": 1.20, "residential": 1.30, "shortcut": 1.80},
        "shortcut_flood_trigger": 1,
        "shortcut_flood_mult": 3.50,
        "notes": "India's worst flooding. Narrowest lanes. Even drizzle floods shortcuts.",
    },
}

ALL_CITIES = list(CITIES.keys())


def get_city(name: str) -> dict:
    key = name.lower().strip()
    if key not in CITIES:
        raise ValueError(f"Unknown city '{name}'. Available: {ALL_CITIES}")
    return CITIES[key]


def list_cities():
    print(f"\n{'City':<12} {'RoadFactor':>10} {'Speed':>7} {'HeavyRain%':>11} "
          f"{'ShortcutFlood':>14}  Notes")
    print("─" * 95)
    for cfg in CITIES.values():
        print(f"  {cfg['name']:<10} {cfg['road_factor']:>10.2f} "
              f"{cfg['v_speed']:>6.0f} "
              f"{cfg['weather_probs'][2]*100:>10.0f}% "
              f"{cfg['route_flood_risks']['shortcut']:>14.2f}  "
              f"{cfg['notes'][:50]}")
    print()


if __name__ == "__main__":
    list_cities()
    print("✓ city_configs.py OK")