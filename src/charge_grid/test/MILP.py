import time
from datetime import datetime
import numpy as np
from pathlib import Path
from ortools.linear_solver import pywraplp

from charge_grid.utils import read_input

def solve_battery_stations_ortools(input_file, alpha=100.0, beta=1.0, lambda_weight=1.0, time_limit_ms=None):
    N, B, C, P, L, R, Z, D = read_input(input_file)
    
    start_time_obj = datetime.now()
    start_time_str = start_time_obj.strftime("%Y-%m-%d %H:%M:%S")
    start_cpu = time.time()

    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise Exception("SCIP solver not available.")
    
    if time_limit_ms:
        solver.SetTimeLimit(time_limit_ms)

    x = [solver.IntVar(0, 1, f"x_{j}") for j in range(N)]
    
    F = [[solver.IntVar(0, solver.infinity(), f"F_{i}_{j}") for j in range(N)] for i in range(N)]

    revenue = solver.Sum(P * F[i][j] for i in range(N) for j in range(N))
    station_cost = solver.Sum(x[j] * (C + R[j]) for j in range(N))
    
    total_demand = sum(D)
    total_flow = solver.Sum(F[i][j] for i in range(N) for j in range(N))
    unmet_penalty = alpha * (total_demand - total_flow)
    
    distance_penalty = solver.Sum(beta * L[i][j] * F[i][j] for i in range(N) for j in range(N))

    E_expr = revenue - station_cost
    O_expr = unmet_penalty + distance_penalty
    
    solver.Maximize(lambda_weight * E_expr - O_expr)

    for i in range(N):
        solver.Add(solver.Sum(F[i][j] for j in range(N)) <= int(D[i]))
        
        for j in range(N):
            if L[i][j] > Z[i]:
                solver.Add(F[i][j] == 0)
                
            solver.Add(F[i][j] <= int(D[i]) * x[j])

    for j in range(N):
        solver.Add(solver.Sum(F[i][j] for i in range(N)) <= int(B) * x[j])

    status = solver.Solve()
    
    end_cpu = time.time()
    end_time_obj = datetime.now()
    end_time_str = end_time_obj.strftime("%Y-%m-%d %H:%M:%S")
    run_time = end_cpu - start_cpu

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        best_x = [int(x[j].solution_value()) for j in range(N)]
        num_stations = sum(best_x)
        
        val_revenue = sum(P * F[i][j].solution_value() for i in range(N) for j in range(N))
        val_cost = sum(best_x[j] * (C + R[j]) for j in range(N))
        val_E = val_revenue - val_cost
        
        val_unmet = sum(D[i] - sum(F[i][j].solution_value() for j in range(N)) for i in range(N))
        val_unmet_pen = alpha * val_unmet
        val_dist = sum(beta * L[i][j] * F[i][j].solution_value() for i in range(N) for j in range(N))
        val_O = val_unmet_pen + val_dist
        
        val_fitness = solver.Objective().Value()

        print(f"Gen MILP | Stations: {num_stations:03d} | Fit: {val_fitness:,.2f} | "
              f"E: {val_E:,.2f} (Rev: {val_revenue:,.2f}, Cost: {val_cost:,.2f}) | "
              f"O: {val_O:,.2f} (Unmet: {val_unmet_pen:,.2f}, Dist: {val_dist:,.2f})\n")
        
        print("--- Optimization Complete ---")
        print(f"Start Time: {start_time_str}")
        print(f"End Time: {end_time_str}")
        print(f"Run Time: {run_time:.2f} seconds")
        print(f"Optimal Station Locations (x): {best_x}")
        print(f"Optimal Fitness Found: {val_fitness:,.2f}")

        return best_x, val_fitness
    else:
        print("The solver could not find an optimal solution.")
        return None, None

if __name__ == "__main__":
    solve_battery_stations_ortools(
        input_file="../input_q1.txt", 
        alpha=10.0, 
        beta=0.0005, 
        lambda_weight=1.0,
        time_limit_ms=2000000 # 2000 seconds
    )