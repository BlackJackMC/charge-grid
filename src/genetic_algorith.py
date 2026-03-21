import pygad
import random
import json
from pathlib import Path
from ortools.linear_solver import pywraplp
from datetime import datetime

data_folder = Path('..')
input_path = data_folder / 'input_q1.txt'

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

def greedy_routing(x, N, B, L, Z, D, alpha: float = 1, beta: float = 1) -> tuple[float, list[list[int]], list[list[int]]]:
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

    F = [[0 for j in range(N)] for i in range(N)]
    y = [[0 for j in range(N)] for i in range(N)]

    for i in range(len(D)):
        for j, dist in nearest_stations[i]:
            if station_battery[j] > 0:
                y[i][j] = 1
                F[i][j] = min(D[i], station_battery[j])
                D[i] -= F[i][j]
                station_battery[j] -= F[i][j]

    loss = alpha * sum(D)

    for i in range(N):
        for j in range(N):
            loss += beta * F[i][j] * L[i][j]

    return loss, F, y

def milp_routing(x, N, B, L, Z, D, alpha: float = 1, beta: float = 1) -> tuple[float, list[list[float]], list[list[int]]]:
    solver = pywraplp.Solver.CreateSolver('SCIP')
    A = [(i, j) for i in range(N) for j in range(N) if L[i][j] <= Z[i]]

    F = {}
    y = {}
    for i, j in A:
        F[i, j] = solver.NumVar(0.0, solver.infinity(), f'F_{i}_{j}')
        y[i, j] = solver.IntVar(0, 1, f'y_{i}_{j}')

    objective_terms = []
    for i in range(N):
        served_i = solver.Sum([F[i, j] for j in range(N) if (i, j) in A])
        objective_terms.append(alpha * (D[i] - served_i))
        
    for i, j in A:
        objective_terms.append(beta * F[i, j] * L[i][j])
        
    solver.Minimize(solver.Sum(objective_terms))

    for j in range(N):
        solver.Add(solver.Sum([F[i, j] for i in range(N) if (i, j) in A]) <= B * x[j])

    for i, j in A:
        solver.Add(F[i, j] <= D[i] * y[i, j])

    for i, j in A:
        for k in range(N):
            if (i, k) in A and L[i][k] < L[i][j]:
                remaining_cap_k = (B * x[k]) - solver.Sum([F[m, k] for m in range(N) if (m, k) in A])
                solver.Add(remaining_cap_k <= B * (2 - y[i, j] - x[k]))

    for i, j in A:
        solver.Add(y[i, j] <= x[j])
        
    for i in range(N):
        solver.Add(solver.Sum([F[i, j] for j in range(N) if (i, j) in A]) <= D[i])

    status = solver.Solve()

    F_out = [[0.0 for j in range(N)] for i in range(N)]
    y_out = [[0 for j in range(N)] for i in range(N)]
    loss = float('inf')

    if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        loss = solver.Objective().Value()
        for i, j in A:
            F_out[i][j] = F[i, j].solution_value()
            y_out[i][j] = int(round(y[i, j].solution_value()))

    return loss, F_out, y_out

def profit(x, N, B, C, P, L, R, Z, D, alpha, beta, routing):
    _, F, _ = routing(x, N, B, L, Z, D, alpha, beta)

    profit_val = 0
    for i in range(N):
        for j in range(N):
            profit_val += x[j] * F[i][j] * P

    for j in range(N):
        profit_val -= x[j] * (C + R[j])

    return profit_val

if __name__ == "__main__":
    N, B, C, P, L, R, Z, D = read_input()
    
    ALPHA = 1.0
    BETA = 1.0
    BEHAVIOR_MODEL = greedy_routing 
    
    GA_NUM_GENERATIONS = 50
    GA_SOL_PER_POP = 20
    GA_NUM_PARENTS_MATING = 10
    GA_MUTATION_PERCENT = 10
    
    generation_history = []

    def fitness_func(ga_instance, solution, solution_idx):
        x = [int(val) for val in solution]
        return profit(x, N, B, C, P, L, R, Z, list(D), ALPHA, BETA, BEHAVIOR_MODEL)

    def on_generation(ga_instance):
        best_sol, best_fit, _ = ga_instance.best_solution()
        current_x = [int(val) for val in best_sol]
        _, current_F, _ = BEHAVIOR_MODEL(current_x, N, B, L, Z, list(D), ALPHA, BETA)
        
        generation_history.append({
            "generation": ga_instance.generations_completed,
            "best_fitness": best_fit,
            "x": current_x,
            "F": current_F
        })
        print(f"Generation {ga_instance.generations_completed} | Best Profit: {best_fit:,.2f}")

    ga_instance = pygad.GA(
        num_generations=GA_NUM_GENERATIONS,
        num_parents_mating=GA_NUM_PARENTS_MATING,
        fitness_func=fitness_func,
        sol_per_pop=GA_SOL_PER_POP,
        num_genes=N,
        gene_space=[0, 1],
        mutation_percent_genes=GA_MUTATION_PERCENT,
        on_generation=on_generation
    )

    print(f"--- Starting Optimization ({BEHAVIOR_MODEL.__name__} lower-level) ---")
    ga_instance.run()

    best_x, best_profit, _ = ga_instance.best_solution()
    best_x = [int(val) for val in best_x]

    print("\n--- Optimization Complete ---")
    print(f"Optimal Station Locations (x): {best_x}")
    print(f"Optimal Profit Found: {best_profit:,.2f}")

    final_loss, final_F, final_y = BEHAVIOR_MODEL(best_x, N, B, L, Z, list(D), ALPHA, BETA)

    final_population_data = []
    if hasattr(ga_instance, "population") and hasattr(ga_instance, "last_generation_fitness"):
        for chrom, fit in zip(ga_instance.population, ga_instance.last_generation_fitness):
            chrom_x = [int(g) for g in chrom]
            _, chrom_F, _ = BEHAVIOR_MODEL(chrom_x, N, B, L, Z, list(D), ALPHA, BETA)
            final_population_data.append({
                "x": chrom_x,
                "fitness": float(fit),
                "F": chrom_F
            })

    output_data = {
        "metadata": {
            "input_file": str(input_path.name),
            "lower_level_model": BEHAVIOR_MODEL.__name__,
            "ga_parameters": {
                "num_generations": GA_NUM_GENERATIONS,
                "sol_per_pop": GA_SOL_PER_POP,
                "num_parents_mating": GA_NUM_PARENTS_MATING,
                "mutation_percent_genes": GA_MUTATION_PERCENT
            },
            "problem_constants": {
                "N": N, "B": B, "C": C, "P": P, "ALPHA": ALPHA, "BETA": BETA
            }
        },
        "best_solution": {
            "x": best_x,
            "E": best_profit,
            "O": final_loss,
            "F": final_F,
            "y": final_y
        },
        "generation_history": generation_history,
        "final_population": final_population_data
    }
    
    output_folder = Path('./output')
    output_folder.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_folder / f'solution_{input_path.name.replace(".txt", "")}_{timestamp}.json'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Detailed results successfully saved to: {output_path}")