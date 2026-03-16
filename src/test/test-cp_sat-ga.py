from ortools.sat.python import cp_model
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
    model = cp_model.CpModel()

    f = [[[model.NewIntVar(0, bat_cap, f"f_{t}_{i}_{j}") 
        for j in range(n)] 
        for i in range(n)] 
        for t in range(T)]

    I = [[model.NewIntVar(0, int(bat_cap), f"I_{t}_{j}") 
        for j in range(n)] 
        for t in range(T)]

    for t in range(T):
        for j in range(n):
            model.Add(sum(f[t][i][j] for i in range(n)) <= I[t][j])

    for t in range(1, T):
        for j in range(n):
            outflow_prev = sum(f[t-1][i][j] for i in range(n))
            
            theoretical_I = model.NewIntVar(0, int(bat_cap + recharge), f"temp_I_{t}_{j}")
            model.Add(theoretical_I == I[t-1][j] - outflow_prev + int(recharge))
            
            model.AddMinEquality(I[t][j], [theoretical_I, int(bat_cap)])

    for j in range(n):
        model.Add(I[0][j] == int(bat_cap))

    for t in range(T):
        for i in range(n):
            model.Add(sum(f[t][i][j] for j in range(n)) <= int(demand[i]))
            for j in range(n):
                model.Add(f[t][i][j] <= x[j] * int(demand[i]))

    model.Minimize(sum([
        alpha * (int(demand[i]) - sum(f[t][i][j] for j in range(n))) + 
        beta * sum(f[t][i][j] * int(dist[i][j]) for j in range(n)) 
        for i in range(n) for t in range(T)
    ]))

    solver = cp_model.CpSolver()
    
    solver.parameters.max_time_in_seconds = 10.0 
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        total_f_val = sum(solver.Value(f[t][i][j]) for t in range(T) for i in range(n) for j in range(n))
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
This stucked at somewhere idk
'''