from ortools.linear_solver import pywraplp
import pygad
import numpy as np

# Read input

with open("../../data/district1.txt") as f:
    n, m, bat_cap, T = [int(i) for i in f.readline().strip().split()] # |V|, |E|, S, T

    build_cost, recharge, profit = [int(i) for i in f.readline().strip().split()] # C, E, P

    # edge_list = [tuple([int(i) for i in f.readline().strip().split()]) for _ in range(m)] # E
    
    dist = np.ndarray((n, n), dtype=np.float32)

    for i in range(m):
        u, v, cost = f.readline().strip().split()
        
        u = int(u) - 1
        v = int(v) - 1
        cost = float(cost)

        dist[u, v] = int(cost * 1000)
        dist[v, u] = int(cost * 1000)
        dist[u, u] = 0

    demand = [int(i) for i in f.readline().strip().split()] # D

# Hyperparameter
alpha = 1
beta = 1
eval_count = 0

def _fitness(x):
    global eval_count
    eval_count += 1
    print(f"Đang đánh giá cá thể thứ {eval_count} / {50 * 30}...", end="\r")

    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        return -10**8

    
    f = [[[None for j in range(n)] for i in range(n)] for t in range(T)]
    I = [[None for j in range(n)] for t in range(T)]

    for j in range(n):
        if x[j] == 0:
            for t in range(T):
                I[t][j] = 0
                for i in range(n):
                    f[t][i][j] = 0
        else:
            for t in range(T):
                
                I[t][j] = solver.NumVar(0, bat_cap, f"I_{t}_{j}")
                for i in range(n):
                    f[t][i][j] = solver.NumVar(0, bat_cap, f"f_{t}_{i}_{j}")

    
    for t in range(T):
        for j in range(n):
            if x[j] == 1:
                solver.Add(sum(f[t][i][j] for i in range(n)) <= I[t][j])
                for i in range(n):
                    solver.Add(f[t][i][j] <= int(demand[i]))

    
    for t in range(1, T):
        for j in range(n):
            if x[j] == 1:
                outflow_prev = sum(f[t-1][i][j] for i in range(n))
                
                solver.Add(I[t][j] <= I[t-1][j] - outflow_prev + int(recharge))
                solver.Add(I[t][j] <= int(bat_cap))

    for j in range(n):
        if x[j] == 1:
            solver.Add(I[0][j] == int(bat_cap))

    for t in range(T):
        for i in range(n):
            solver.Add(sum(f[t][i][j] for j in range(n) if x[j] == 1) <= int(demand[i]))

    penalty_terms = []
    for t in range(T):
        for i in range(n):
            served_demand = sum(f[t][i][j] for j in range(n) if x[j] == 1)
            penalty_terms.append(alpha * (int(demand[i]) - served_demand))
            
            for j in range(n):
                if x[j] == 1:
                    penalty_terms.append(beta * f[t][i][j] * int(dist[i][j]))
                    
    solver.Minimize(solver.Sum(penalty_terms))

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        total_f_val = 0
        for t in range(T):
            for i in range(n):
                for j in range(n):
                    if x[j] == 1:
                        total_f_val += f[t][i][j].solution_value()
                        
        return profit * total_f_val - build_cost * sum(x)
    else:
        return -10**8
    
def fitness(ga_instance, solution, solution_idx):
    return _fitness(solution)

def on_generation(ga_instance):
    current_gen = ga_instance.generations_completed
    
    best_fitness = ga_instance.best_solution()[1]
    
    print(f"Thế hệ {current_gen:02d} | Fitness tốt nhất: {best_fitness:,.2f}")

ga_instance = pygad.GA(
    num_generations=50,
    num_parents_mating=10,
    fitness_func=fitness,
    sol_per_pop=30,
    num_genes=n,
    gene_type=int,
    gene_space=[0, 1],
    mutation_probability=0.1,
    on_generation=on_generation
)

ga_instance.run()
solution, solution_fitness, _ = ga_instance.best_solution()
print(f"Cấu hình trạm tối ưu nhất (x): {solution}")
print(f"Lợi nhuận Z cao nhất đạt được: {solution_fitness}")

'''
Thế hệ 01 | Fitness tốt nhất: 74,280.00
Thế hệ 02 | Fitness tốt nhất: 76,085.00
Thế hệ 03 | Fitness tốt nhất: 78,035.00
Thế hệ 04 | Fitness tốt nhất: 78,280.00
Thế hệ 05 | Fitness tốt nhất: 83,565.00
Thế hệ 06 | Fitness tốt nhất: 83,565.00
Thế hệ 07 | Fitness tốt nhất: 83,565.00
Thế hệ 08 | Fitness tốt nhất: 84,370.00
Thế hệ 09 | Fitness tốt nhất: 84,370.00
Thế hệ 10 | Fitness tốt nhất: 84,440.00
Thế hệ 11 | Fitness tốt nhất: 84,440.00
Thế hệ 12 | Fitness tốt nhất: 84,440.00
Thế hệ 13 | Fitness tốt nhất: 84,440.00
Thế hệ 14 | Fitness tốt nhất: 84,440.00
Thế hệ 15 | Fitness tốt nhất: 84,440.00
Thế hệ 16 | Fitness tốt nhất: 87,520.00
Thế hệ 17 | Fitness tốt nhất: 87,520.00
Thế hệ 18 | Fitness tốt nhất: 87,520.00
Thế hệ 19 | Fitness tốt nhất: 90,360.00
Thế hệ 20 | Fitness tốt nhất: 90,360.00
Thế hệ 21 | Fitness tốt nhất: 90,360.00
Thế hệ 22 | Fitness tốt nhất: 90,360.00
Thế hệ 23 | Fitness tốt nhất: 90,360.00
Thế hệ 24 | Fitness tốt nhất: 90,360.00
Thế hệ 25 | Fitness tốt nhất: 90,360.00
Thế hệ 26 | Fitness tốt nhất: 90,360.00
Thế hệ 27 | Fitness tốt nhất: 90,360.00
Thế hệ 28 | Fitness tốt nhất: 90,360.00
Thế hệ 29 | Fitness tốt nhất: 90,360.00
Thế hệ 30 | Fitness tốt nhất: 90,360.00
Thế hệ 31 | Fitness tốt nhất: 90,360.00
Thế hệ 32 | Fitness tốt nhất: 90,360.00
Thế hệ 33 | Fitness tốt nhất: 90,360.00
Thế hệ 34 | Fitness tốt nhất: 90,360.00
Thế hệ 35 | Fitness tốt nhất: 90,360.00
Thế hệ 36 | Fitness tốt nhất: 90,360.00
Thế hệ 37 | Fitness tốt nhất: 90,360.00
Thế hệ 38 | Fitness tốt nhất: 90,360.00
Thế hệ 39 | Fitness tốt nhất: 90,360.00
Thế hệ 40 | Fitness tốt nhất: 90,360.00
Thế hệ 41 | Fitness tốt nhất: 90,360.00
Thế hệ 42 | Fitness tốt nhất: 90,835.00
Thế hệ 43 | Fitness tốt nhất: 90,835.00
Thế hệ 44 | Fitness tốt nhất: 90,835.00
Thế hệ 45 | Fitness tốt nhất: 92,445.00
Thế hệ 46 | Fitness tốt nhất: 92,445.00
Thế hệ 47 | Fitness tốt nhất: 92,445.00
Thế hệ 48 | Fitness tốt nhất: 92,445.00
Thế hệ 49 | Fitness tốt nhất: 92,445.00
Thế hệ 50 | Fitness tốt nhất: 92,445.00
Cấu hình trạm tối ưu nhất (x): [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1
, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1
, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0
, 0, 1, 1, 1]
Lợi nhuận Z cao nhất đạt được: 92445.0
'''