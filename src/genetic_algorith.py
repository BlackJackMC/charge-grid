import pygad
import random
import json
from pathlib import Path
from datetime import datetime

input_folder = Path('..')
output_folder = Path('./output')
input_path = input_folder / 'input_q1.txt'

#### I/O functions

def read_input():
    if input_path.exists():
        with open(input_path, 'r', encoding='utf-8') as f:
            N, B, C, P = f.readline().strip().split()
            N = int(N)
            B = int(B)
            C = float(C)
            P = float(P)

            L = [[float(x) for x in f.readline().strip().split()] for _ in range(N)]
            R = [float(x) for x in f.readline().strip().split()]
            Z = [float(x) for x in f.readline().strip().split()]
            D = [int(x) for x in f.readline().strip().split()]

        return N, B, C, P, L, R, Z, D

def save_optimization_results(best_x, best_fitness, generation_history, config, input_filename):
    clean_config = {k: (v.__name__ if callable(v) else v) for k, v in config.items()}

    output_data = {
        "metadata": {
            "input_file": input_filename,
            "configuration": clean_config
        },
        "best_solution": {
            "x": best_x,
            "fitness_score": best_fitness
        },
        "generation_history": generation_history
    }
    
    output_folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_folder / f'solution_{input_filename.replace(".txt", "")}_{timestamp}.json'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Detailed results successfully saved to: {output_path}")

###################
#### Core Logic

def route(x, demand_order=None) -> list[list[int]]:
    station_battery = {i: B for i in range(N) if x[i] == 1}

    nearest_stations = {
        i: [
            j
            for j in filter(
                lambda k: k[1] <= Z[i] and x[k[0]] == 1,
                sorted(enumerate(l), key=lambda k: k[1])
            )
        ]
        for i, l in enumerate(L)
    }

    F = [[0 for _ in range(N)] for _ in range(N)]

    if demand_order is None:
        demand_order = list(range(len(D)))
        random.shuffle(demand_order)

    local_D = list(D)

    for i in demand_order:
        for j, dist in nearest_stations[i]:
            if station_battery[j] > 0:
                F[i][j] = min(local_D[i], station_battery[j])
                local_D[i] -= F[i][j]
                station_battery[j] -= F[i][j]

    return F

def E(x, F):
    profit_val = 0
    for i in range(N):
        for j in range(N):
            profit_val += x[j] * F[i][j] * P

    for j in range(N):
        profit_val -= x[j] * (C + R[j])

    return profit_val

def O(F, alpha: float = 1, beta: float = 1):
    dissatisfaction = 0
    for i in range(N):
        dissatisfaction += alpha * (D[i] - sum(F[i]))
        for j in range(N):
            dissatisfaction += beta * F[i][j] * L[i][j]

    return dissatisfaction

def fitness(x):
    fitness_vals = []
    
    for order in evaluation_orders:
        F = config['behavior_model'](x, demand_order=order)
        fit = config['lambda'] * E(x, F) - config['mu'] * O(F, config['alpha'], config['beta'])
        fitness_vals.append(fit)
        
    return sum(fitness_vals) / config['num_shuffles']

###################
#### GA callback & execution functions

def fitness_handler(ga_instance, solution, solution_idx):
    x = [int(val) for val in solution]
    return fitness(x)

def log_handler(ga_instance):
    best_sol, best_fit, _ = ga_instance.best_solution()
    current_x = [int(val) for val in best_sol]
    
    generation_history.append({
        "generation": ga_instance.generations_completed,
        "best_average_fitness": best_fit,
        "x": current_x
    })
    
    print(f"Generation {ga_instance.generations_completed:02d} | Fitness: {best_fit:,.2f}")

def run_optimization(ga_instance, model_name):
    print(f"--- Starting Optimization ({model_name}) ---")
    ga_instance.run()

    best_x, best_fitness, _ = ga_instance.best_solution()
    best_x = [int(val) for val in best_x]

    print("\n--- Optimization Complete ---")
    print(f"Optimal Station Locations (x): {best_x}")
    print(f"Optimal Fitness Found: {best_fitness:,.2f}")
    
    return best_x, best_fitness

###################

if __name__ == "__main__": 
    problem_data = read_input()
    N, B, C, P, L, R, Z, D = problem_data
    
    config = {
        'alpha': 10.0,
        'beta': 1.0,
        'lambda': 1.0,
        'mu': 1.0,
        'behavior_model': route,
        'num_generations': 50,
        'sol_per_pop': 20,
        'num_parents_mating': 10,
        'mutation_percent_genes': 10,
        'num_shuffles': 5,
        'random_seed': 42
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
        gene_space=[0, 1],
        mutation_percent_genes=config['mutation_percent_genes'],
        on_generation=log_handler,
        random_seed=config['random_seed']
    )

    best_x, best_fitness = run_optimization(ga_instance, config['behavior_model'].__name__)

    save_optimization_results(
        best_x, 
        best_fitness, 
        generation_history, 
        config, 
        input_path.name
    )