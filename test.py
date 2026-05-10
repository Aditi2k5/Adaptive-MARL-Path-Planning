"""
Complete System Test
Tests all components together and compares baselines
"""

import numpy as np
from environment import QuickCommerceEnv
from assignment import (random_assignment, greedy_urgent_assignment,
                       random_route, greedy_nearest_route, urgent_first_route)
import config


def test_single_episode(strategy_name: str, assignment_func, routing_func):
    """
    Test one complete episode with given strategy
    
    Args:
        strategy_name: Name for display
        assignment_func: Function for order assignment
        routing_func: Function for route planning
        
    Returns:
        Episode statistics
    """
    print(f"\n{'='*60}")
    print(f"TESTING: {strategy_name}")
    print(f"{'='*60}")
    
    env = QuickCommerceEnv()
    state = env.reset()
    
    env.print_status()
    
    # Get pending orders
    pending = env.get_pending_orders()
    
    if not pending:
        print("No orders to process!")
        return None
    
    # Assignment stage
    print(f"\nAssigning {len(pending)} orders...")
    assignments = assignment_func(pending, env.riders, env.current_time)
    
    # Group orders by rider
    rider_orders = {}
    for order_id, rider_id in assignments.items():
        if rider_id not in rider_orders:
            rider_orders[rider_id] = []
        rider_orders[rider_id].append(order_id)
    
    print(f"  Assigned to {len(rider_orders)} riders")
    
    # Routing stage
    print(f"\nPlanning routes...")
    for rider_id, order_ids in rider_orders.items():
        # Assign orders to rider
        for order_id in order_ids:
            env.assign_order(order_id, rider_id)
        
        # Plan route
        if strategy_name == "Random Baseline":
            route = routing_func(order_ids)
        elif "Greedy Nearest" in strategy_name:
            route = routing_func(order_ids, env.orders,
                               env.riders[rider_id].location, env.network)
        elif "Urgent First" in strategy_name:
            route = routing_func(order_ids, env.orders, env.current_time)
        else:
            route = order_ids  # Simple order
        
        print(f"  Rider {rider_id}: route {route}")
        
        # Execute route
        dist, time = env.execute_route(rider_id, route)
        print(f"    Distance: {dist:.2f} km, Time: {time:.1f} min")
    
    # Summary
    env.print_summary()
    
    # Collect statistics
    stats = {
        'total_orders': len(env.orders),
        'delivered': env.completed_deliveries,
        'on_time': len([o for o in env.orders.values() if o.delivered and not o.is_late()]),
        'late': len([o for o in env.orders.values() if o.delivered and o.is_late()]),
        'total_distance': sum(r.total_distance for r in env.riders),
        'cost_J': env.compute_cost_J()
    }
    
    return stats


def compare_strategies(n_runs=10):
    """
    Compare multiple strategies over multiple episodes
    
    Args:
        n_runs: Number of episodes per strategy
    """
    print("\n" + "="*60)
    print(f"COMPARING STRATEGIES ({n_runs} episodes each)")
    print("="*60)
    
    strategies = [
        {
            'name': 'Random Baseline',
            'assignment': lambda o, r, t: random_assignment(o, r),
            'routing': random_route
        },
        {
            'name': 'Greedy Urgent + Random Route',
            'assignment': greedy_urgent_assignment,
            'routing': random_route
        },
        {
            'name': 'Greedy Urgent + Nearest Route',
            'assignment': greedy_urgent_assignment,
            'routing': greedy_nearest_route
        },
        {
            'name': 'Greedy Urgent + Urgent First Route',
            'assignment': greedy_urgent_assignment,
            'routing': urgent_first_route
        }
    ]
    
    results = {s['name']: {
        'costs': [],
        'on_time_rates': [],
        'distances': []
    } for s in strategies}
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Running: {strategy['name']}")
        print(f"{'='*60}")
        
        for run in range(n_runs):
            # Create environment
            env = QuickCommerceEnv()
            env.reset()
            
            # Get pending orders
            pending = env.get_pending_orders()
            
            if not pending:
                continue
            
            # Assignment
            assignments = strategy['assignment'](pending, env.riders, env.current_time)
            
            # Group by rider
            rider_orders = {}
            for order_id, rider_id in assignments.items():
                if rider_id not in rider_orders:
                    rider_orders[rider_id] = []
                rider_orders[rider_id].append(order_id)
            
            # Routing and execution
            for rider_id, order_ids in rider_orders.items():
                for order_id in order_ids:
                    env.assign_order(order_id, rider_id)
                
                # Route based on strategy
                if strategy['name'] == 'Random Baseline':
                    route = strategy['routing'](order_ids)
                elif 'Nearest Route' in strategy['name']:
                    route = strategy['routing'](order_ids, env.orders,
                                               env.riders[rider_id].location, env.network)
                elif 'Urgent First' in strategy['name']:
                    route = strategy['routing'](order_ids, env.orders, env.current_time)
                else:
                    route = order_ids
                
                env.execute_route(rider_id, route)
            
            # Collect metrics
            delivered = env.completed_deliveries
            if delivered > 0:
                on_time = len([o for o in env.orders.values() if o.delivered and not o.is_late()])
                on_time_rate = on_time / delivered
            else:
                on_time_rate = 0
            
            total_distance = sum(r.total_distance for r in env.riders)
            cost = env.compute_cost_J()
            
            results[strategy['name']]['costs'].append(cost)
            results[strategy['name']]['on_time_rates'].append(on_time_rate)
            results[strategy['name']]['distances'].append(total_distance)
            
            print(f"  Run {run+1}: Cost=₹{cost:.2f}, On-time={on_time_rate*100:.1f}%")
    
    # Print comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    
    print(f"\n{'Strategy':<40} {'Cost J (₹)':<15} {'On-time %':<12} {'Distance (km)'}")
    print("-" * 80)
    
    baseline_cost = None
    
    for strategy in strategies:
        name = strategy['name']
        costs = results[name]['costs']
        on_time = results[name]['on_time_rates']
        distances = results[name]['distances']
        
        if costs:
            avg_cost = np.mean(costs)
            std_cost = np.std(costs)
            avg_on_time = np.mean(on_time) * 100
            avg_dist = np.mean(distances)
            
            if baseline_cost is None:
                baseline_cost = avg_cost
            
            improvement = (baseline_cost - avg_cost) / baseline_cost * 100
            
            print(f"{name:<40} {avg_cost:>6.2f}±{std_cost:>4.2f}  "
                  f"{avg_on_time:>5.1f}%  {avg_dist:>8.2f}  "
                  f"({improvement:+.1f}%)")
    
    print("=" * 80)
    
    # Best strategy
    best_strategy = min(results.keys(), key=lambda k: np.mean(results[k]['costs']))
    print(f"\n✓ Best strategy: {best_strategy}")
    print(f"  Avg cost: ₹{np.mean(results[best_strategy]['costs']):.2f}")
    print(f"  On-time rate: {np.mean(results[best_strategy]['on_time_rates'])*100:.1f}%")


def quick_demo():
    """Quick demonstration of the system"""
    print("\n" + "="*60)
    print("QUICK DEMO - Your MARL System in Action")
    print("="*60)
    
    # Test with greedy urgent + nearest route (best baseline)
    stats = test_single_episode(
        "Greedy Urgent + Nearest Route",
        greedy_urgent_assignment,
        greedy_nearest_route
    )
    
    if stats:
        print("\n" + "="*60)
        print("KEY RESULTS:")
        print("="*60)
        print(f"✓ Delivered: {stats['delivered']}/{stats['total_orders']} orders")
        print(f"✓ On-time: {stats['on_time']} ({stats['on_time']/max(1,stats['delivered'])*100:.1f}%)")
        print(f"✓ Total distance: {stats['total_distance']:.2f} km")
        print(f"✓ COST J (your objective): ₹{stats['cost_J']:.2f}")
        print("="*60)


if __name__ == "__main__":
    print("="*60)
    print("QUICK COMMERCE MARL - COMPLETE SYSTEM TEST")
    print("="*60)
    
    # Quick demo
    quick_demo()
    
    # Full comparison
    print("\n\nRunning full baseline comparison...")
    compare_strategies(n_runs=10)
    
    print("\n" + "="*60)
    print("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nYour system is working!")
    print("Next steps:")
    print("  1. ✓ Baselines working (you have this now)")
    print("  2. Add RL agent to beat baselines")
    print("  3. Add MARL for multi-agent coordination")
    print("  4. Generate plots for paper")