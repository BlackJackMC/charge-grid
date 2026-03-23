import pygad
import random
import json
import numpy as np
from pathlib import Path
from datetime import datetime

input_folder = Path('..')
output_folder = Path('./output')
input_path = input_folder / 'input_q1.txt'

def read_input():
    if input_path.exists():
        with open(input_path, 'r', encoding='utf-8') as f:
            N, B, C, P = f.readline().strip().split()
            N, B = int(N), int(B)
            C, P = float(C), float(P)

            L = np.array([[float(x) for x in f.readline().strip().split()] for _ in range(N)])
            R = np.array([float(x) for x in f.readline().strip().split()])
            Z = np.array([float(x) for x in f.readline().strip().split()])
            D = np.array([float(x) for x in f.readline().strip().split()])

        return N, B, C, P, L, R, Z, D
    raise FileNotFoundError(f"Input file not found at {input_path}")

def save_optimization_results(best_x, best_fitness, generation_history, config, input_filename, start_time, end_time, run_time):
    clean_config = {k: (v.__name__ if callable(v) else v) for k, v in config.items()}

    output_data = {
        "metadata": {
            "input_file": input_filename,
            "configuration": clean_config
        },
        "timing": {
            "start_time": start_time,
            "end_time": end_time,
            "run_time_seconds": run_time
        },
        "best_solution": {
            "fitness_score": float(best_fitness),
            "x": best_x,
        },
        "generation_history": generation_history
    }
    
    output_folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_folder / f'solution_{input_filename.replace(".txt", "")}_{timestamp}.json'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Detailed results successfully saved to: {output_path}")

def route(x, station_order=None) -> np.ndarray:
    station_battery = {j: B for j in range(N) if x[j] == 1}
    F = np.zeros((N, N))

    if station_order is None:
        station_order = list(range(N))
        random.shuffle(station_order)

    local_D = D.copy()

    for j in station_order:
        if x[j] == 1:
            for i in precomputed_nearest[j]:
                if local_D[i] > 0 and station_battery[j] > 0:
                    served = min(local_D[i], station_battery[j])
                    F[i, j] = served
                    local_D[i] -= served
                    station_battery[j] -= served
                if station_battery[j] == 0:
                    break
    return F

def E(x, F):
    revenue = np.sum(F * x) * P
    cost = np.sum(x * (C + R))
    total_profit = revenue - cost
    return total_profit, revenue, cost

def O(F, alpha: float, beta: float):
    unmet_penalty = np.sum(D - np.sum(F, axis=1))
    distance_penalty = np.sum(F * L)
    total_dissatisfaction = alpha * unmet_penalty + beta * distance_penalty
    return total_dissatisfaction, unmet_penalty, distance_penalty

def fitness(x):
    fitness_vals = np.zeros(config['num_shuffles'])
    
    for idx, order in enumerate(evaluation_orders):
        F = config['behavior_model'](x, station_order=order)
        total_E, _, _ = E(x, F)
        total_O, _, _ = O(F, config['alpha'], config['beta'])
        fitness_vals[idx] = config['lambda'] * total_E - total_O
        
    return np.mean(fitness_vals)

def fitness_handler(ga_instance, solution, solution_idx):
    return fitness(solution)

def log_handler(ga_instance):
    best_sol, best_fit, _ = ga_instance.best_solution()
    
    e_vals, rev_vals, cost_vals = [], [], []
    o_vals, unmet_vals, dist_vals = [], [], []
    
    for order in evaluation_orders:
        F = config['behavior_model'](best_sol, station_order=order)
        total_E, rev, cost = E(best_sol, F)
        total_O, unmet, dist = O(F, config['alpha'], config['beta'])
        
        e_vals.append(total_E)
        rev_vals.append(rev)
        cost_vals.append(cost)
        o_vals.append(total_O)
        unmet_vals.append(unmet)
        dist_vals.append(dist)
        
    avg_E = np.mean(e_vals)
    avg_rev = np.mean(rev_vals)
    avg_cost = np.mean(cost_vals)
    avg_O = np.mean(o_vals)
    avg_unmet = np.mean(unmet_vals)
    avg_dist = np.mean(dist_vals)
    
    generation_history.append({
        "generation": ga_instance.generations_completed,
        "best_average_fitness": float(best_fit),
        "avg_E_profit": float(avg_E),
        "avg_E_revenue": float(avg_rev),
        "avg_E_cost": float(avg_cost),
        "avg_O_loss": float(avg_O),
        "avg_O_unmet_penalty": float(avg_unmet),
        "avg_O_distance_penalty": float(avg_dist),
        "x": best_sol.tolist()
    })
    
    print(f"Gen {ga_instance.generations_completed:02d} | Stations: {int(np.sum(best_sol)):03d} | Fit: {best_fit:,.2f} | "
          f"E: {avg_E:,.2f} (Rev: {avg_rev:,.2f}, Cost: {avg_cost:,.2f}) | "
          f"O: {avg_O:,.2f} (Unmet: {avg_unmet:,.2f}, Dist: {avg_dist:,.2f})")

def run_optimization(ga_instance, model_name):
    print(f"--- Starting Optimization ({model_name}) ---")
    
    start_time = datetime.now()
    ga_instance.run()
    end_time = datetime.now()
    
    run_time = (end_time - start_time).total_seconds()
    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")

    best_x, best_fitness, _ = ga_instance.best_solution()
    best_x = [int(val) for val in best_x]

    print("\n--- Optimization Complete ---")
    print(f"Start Time: {start_time_str}")
    print(f"End Time: {end_time_str}")
    print(f"Run Time: {run_time:.2f} seconds")
    print(f"Optimal Station Locations (x): {best_x}")
    print(f"Optimal Fitness Found: {best_fitness:,.2f}")
    
    return best_x, best_fitness, start_time_str, end_time_str, run_time

if __name__ == "__main__": 
    N, B, C, P, L, R, Z, D = read_input()

    precomputed_nearest = []
    for j in range(N):
        sorted_demands = np.argsort(L[:, j])
        valid_demands = [i for i in sorted_demands if L[i, j] <= Z[i]]
        precomputed_nearest.append(valid_demands)
    
    config = {
        'alpha': 10.0,
        'beta': 0.0005,
        'lambda': 1.0,
        'behavior_model': route,
        'num_generations': 200,
        'sol_per_pop': 100,
        'num_parents_mating': 10,
        'num_shuffles': 5,
        'random_seed': 42,
        'stop_criteria': ['saturate_20'],
        'parent_selection_type': 'tournament',
        'K_tournament': 3,
        'crossover_type': 'uniform',
        'crossover_probability': 0.8,
        'mutation_type': 'adaptive',
        'mutation_probability': [0.25, 0.05],
        'keep_elitism': 2
    }
    
    rng = random.Random(config['random_seed'])

    evaluation_orders = []
    for _ in range(config['num_shuffles']):
        order = list(range(N))
        rng.shuffle(order)
        evaluation_orders.append(order)

    generation_history = []

    ga_instance = pygad.GA(
        num_generations=config['num_generations'],
        num_parents_mating=config['num_parents_mating'],
        fitness_func=fitness_handler,
        sol_per_pop=config['sol_per_pop'],
        num_genes=N,
        gene_type=int,
        gene_space=[0, 1],
        stop_criteria=config['stop_criteria'],
        on_generation=log_handler,
        random_seed=config['random_seed'],
        parent_selection_type=config['parent_selection_type'],
        K_tournament=config['K_tournament'],
        crossover_type=config['crossover_type'],
        crossover_probability=config['crossover_probability'],
        mutation_type=config['mutation_type'],
        mutation_probability=config['mutation_probability'],
        keep_elitism=config['keep_elitism'],
    )

    best_x, best_fitness, start_time, end_time, run_time = run_optimization(ga_instance, config['behavior_model'].__name__)

    save_optimization_results(
        best_x, 
        best_fitness, 
        generation_history, 
        config, 
        input_path.name,
        start_time,
        end_time,
        run_time
    )