import pygad
import random
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from functools import cache

input_folder = Path('..')
output_folder = Path('./output')
input_path = input_folder / 'input_q1.txt'

###################
#### I/O Functions

def read_input():
    if input_path.exists():
        with open(input_path, 'r', encoding='utf-8') as f:
            N, B, C, P = f.readline().strip().split()
            N, B = int(N), int(B)
            C, P = float(C), float(P)

            L = np.array([[float(x) for x in f.readline().strip().split()] for _ in range(N)])
            R = np.array([float(x) for x in f.readline().strip().split()])
            Z = np.array([float(x) for x in f.readline().strip().split()])
            D = np.array([int(x) for x in f.readline().strip().split()])

        return N, B, C, P, L, R, Z, D
    raise FileNotFoundError(f"Input file not found at {input_path}")

def save_optimization_results(best_x, best_fitness, generation_history, config, input_filename, start_time, end_time, run_time, model_name):
    clean_config = {k: (v.__name__ if callable(v) else v) for k, v in config.items()}

    output_data = {
        "metadata": {
            "input_file": input_filename,
            "model_used": model_name,
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

###################
#### Helper Functions

def E(x, F, C, P, R):
    x = np.array(x)
    revenue = np.sum(F * x) * P
    cost = np.sum(x * (C + R))
    return revenue - cost, revenue, cost

def O(F, D, L, alpha, beta):
    unmet = np.sum(D - np.sum(F, axis=1))
    dist = np.sum(F * L)
    unmet_penalty = alpha * unmet
    distance_penalty = beta * dist
    return unmet_penalty + distance_penalty, unmet_penalty, distance_penalty

def custom_intersection_crossover(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    while len(offspring) < offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :]
        parent2 = parents[(idx + 1) % parents.shape[0], :]

        agree_mask = (parent1 == parent2)
        take_from_p1 = np.random.rand(*parent1.shape) < 0.5
        
        child = np.where(agree_mask, parent1, np.where(take_from_p1, parent1, parent2))
        offspring.append(child)
        idx += 1
        
    return np.array(offspring)

###################
#### Models

class BaseModel:
    def __init__(self, N, B, C, P, R, L, Z, D, config):
        self.name = 'Base_Model'
        self.N = N
        self.B = B
        self.C = C
        self.P = P
        self.R = R
        self.L = L
        self.Z = Z
        self.D = D
        self.config = config

    def route(self, x, *args, **kwargs):
        raise NotImplementedError

    def fitness(self, x):
        raise NotImplementedError

    def get_details(self, x):
        raise NotImplementedError

class CustomerRouting(BaseModel):
    def __init__(self, N, B, C, P, R, L, Z, D, config):
        super().__init__(N, B, C, P, R, L, Z, D, config)
        self.name = 'Customer_Centric_Greedy'

        self.precomputed = []
        for i in range(self.N):
            sorted_stations = np.argsort(self.L[i, :])
            valid_stations = [j for j in sorted_stations if self.L[i, j] <= self.Z[i]]
            self.precomputed.append(valid_stations)

        rng = random.Random(self.config['random_seed'])
        self.evaluation_orders = []
        for _ in range(self.config['num_shuffles']):
            order = list(range(self.N))
            rng.shuffle(order)
            self.evaluation_orders.append(order)

    def route(self, x, eval_order):
        station_battery = {j: self.B for j in range(self.N) if x[j] == 1}
        F = np.zeros((self.N, self.N))
        local_D = self.D.copy()

        for i in eval_order:
            if local_D[i] > 0:
                for j in self.precomputed[i]:
                    if x[j] == 1 and station_battery[j] > 0:
                        served = min(local_D[i], station_battery[j])
                        F[i, j] += served
                        local_D[i] -= served
                        station_battery[j] -= served
                        if local_D[i] == 0:
                            break
        return F

    def fitness(self, x):
        fitness_vals = np.zeros(len(self.evaluation_orders))
        for idx, order in enumerate(self.evaluation_orders):
            F = self.route(x, order)
            total_E, _, _ = E(x, F, self.C, self.P, self.R)
            total_O, _, _ = O(F, self.D, self.L, self.config['alpha'], self.config['beta'])
            fitness_vals[idx] = self.config['lambda'] * total_E - self.config.get('mu', 1.0) * total_O
        return np.mean(fitness_vals)

    def get_details(self, x):
        e_vals, rev_vals, cost_vals, o_vals, unmet_vals, dist_vals = [], [], [], [], [], []
        for order in self.evaluation_orders:
            F = self.route(x, order)
            total_E, rev, cost = E(x, F, self.C, self.P, self.R)
            total_O, unmet, dist = O(F, self.D, self.L, self.config['alpha'], self.config['beta'])
            
            e_vals.append(total_E); rev_vals.append(rev); cost_vals.append(cost)
            o_vals.append(total_O); unmet_vals.append(unmet); dist_vals.append(dist)
            
        return {
            'avg_E': np.mean(e_vals), 'avg_rev': np.mean(rev_vals), 'avg_cost': np.mean(cost_vals),
            'avg_O': np.mean(o_vals), 'avg_unmet': np.mean(unmet_vals), 'avg_dist': np.mean(dist_vals)
        }

class StationRouting(BaseModel):
    def __init__(self, N, B, C, P, R, L, Z, D, config):
        super().__init__(N, B, C, P, R, L, Z, D, config)
        self.name = 'Station_Centric_Greedy'

        self.precomputed = []
        for j in range(self.N):
            sorted_demands = np.argsort(self.L[:, j])
            valid_demands = [i for i in sorted_demands if self.L[i, j] <= self.Z[i]]
            self.precomputed.append(valid_demands)

        rng = random.Random(self.config['random_seed'])
        self.evaluation_orders = []
        for _ in range(self.config['num_shuffles']):
            order = list(range(self.N))
            rng.shuffle(order)
            self.evaluation_orders.append(order)

    def route(self, x, eval_order):
        station_battery = {j: self.B for j in range(self.N) if x[j] == 1}
        F = np.zeros((self.N, self.N))
        local_D = self.D.copy()

        for j in eval_order:
            if x[j] == 1:
                for i in self.precomputed[j]:
                    if local_D[i] > 0 and station_battery[j] > 0:
                        served = min(local_D[i], station_battery[j])
                        F[i, j] += served
                        local_D[i] -= served
                        station_battery[j] -= served
                    if station_battery[j] == 0:
                        break
        return F

    def fitness(self, x):
        fitness_vals = np.zeros(len(self.evaluation_orders))
        for idx, order in enumerate(self.evaluation_orders):
            F = self.route(x, order)
            total_E, _, _ = E(x, F, self.C, self.P, self.R)
            total_O, _, _ = O(F, self.D, self.L, self.config['alpha'], self.config['beta'])
            fitness_vals[idx] = self.config['lambda'] * total_E - self.config.get('mu', 1.0) * total_O
        return np.mean(fitness_vals)

    def get_details(self, x):
        e_vals, rev_vals, cost_vals, o_vals, unmet_vals, dist_vals = [], [], [], [], [], []
        for order in self.evaluation_orders:
            F = self.route(x, order)
            total_E, rev, cost = E(x, F, self.C, self.P, self.R)
            total_O, unmet, dist = O(F, self.D, self.L, self.config['alpha'], self.config['beta'])
            
            e_vals.append(total_E); rev_vals.append(rev); cost_vals.append(cost)
            o_vals.append(total_O); unmet_vals.append(unmet); dist_vals.append(dist)
            
        return {
            'avg_E': np.mean(e_vals), 'avg_rev': np.mean(rev_vals), 'avg_cost': np.mean(cost_vals),
            'avg_O': np.mean(o_vals), 'avg_unmet': np.mean(unmet_vals), 'avg_dist': np.mean(dist_vals)
        }

class BehavioralRouting(BaseModel):
    def __init__(self, N, B, C, P, R, L, Z, D, config):
        super().__init__(N, B, C, P, R, L, Z, D, config)
        self.name = 'BehavioralRouting_Model'

    def route(self, x):
        K = self.config['K']
        w1 = self.config['w1']
        w2 = self.config['w2']
        epsilon = self.config['epsilon']
        
        x_arr = np.array(x)
        open_j = np.nonzero(x_arr)[0]
        M = len(open_j)
        
        F = np.zeros((self.N, self.N), dtype=int)
        
        if M == 0 or np.sum(self.D) == 0:
            return F
            
        batt = np.full(M, float(self.B))
        local_D = np.array(self.D, dtype=int)
        chunk_size = np.maximum(1, local_D // K)
        
        L_sub = self.L[:, open_j]
        Z_col = self.Z[:, None]
        
        valid_mask = L_sub <= Z_col
        V = np.sum(local_D[:, None] * valid_mask, axis=0)
        
        norm_dist_all = np.divide(L_sub, Z_col, out=np.zeros_like(L_sub), where=Z_col!=0)
        congestion_all = V / self.B if self.B > 0 else np.zeros(M)
        
        base_term1 = w1 * norm_dist_all
        base_coeff = norm_dist_all * congestion_all

        for step in range(K + 1):
            active_i = np.nonzero(local_D > 0)[0]
            if len(active_i) == 0:
                break
                
            delta_req_active = np.minimum(local_D[active_i], chunk_size[active_i])
            depletion = self.B / (batt + epsilon)
            
            Loss = base_term1[active_i, :] + w2 * np.log1p(base_coeff[active_i, :] * depletion)
            
            L_active = L_sub[active_i, :]
            Z_active = Z_col[active_i, :]
            invalid_mask = (L_active > Z_active) | (batt == 0)
            Loss[invalid_mask] = np.inf
            
            min_loss = np.min(Loss, axis=1)
            best_rel_j = np.argmin(Loss, axis=1)
            
            valid_assignments = min_loss != np.inf
            if not np.any(valid_assignments):
                break
                
            assigned_i = active_i[valid_assignments]
            assigned_rel_j = best_rel_j[valid_assignments]
            reqs = delta_req_active[valid_assignments]
            
            requested_amount = np.zeros(M, dtype=int)
            np.add.at(requested_amount, assigned_rel_j, reqs)
            
            stations_with_reqs = np.nonzero(requested_amount > 0)[0]
            
            for rel_j in stations_with_reqs:
                abs_j = open_j[rel_j]
                req_j = requested_amount[rel_j]
                avail_batt = batt[rel_j]
                
                mask_j = (assigned_rel_j == rel_j)
                cust_i_for_j = assigned_i[mask_j]
                req_for_j = reqs[mask_j]
                
                if req_j <= avail_batt:
                    F[cust_i_for_j, abs_j] += req_for_j
                    local_D[cust_i_for_j] -= req_for_j
                    batt[rel_j] -= req_j
                else:
                    ratio = avail_batt / req_j
                    allocated = (req_for_j * ratio).astype(int)
                    
                    F[cust_i_for_j, abs_j] += allocated
                    local_D[cust_i_for_j] -= allocated
                    
                    leftover = int(avail_batt - np.sum(allocated))
                    batt[rel_j] = 0
                    
                    if leftover > 0:
                        dist_to_j = self.L[cust_i_for_j, abs_j]
                        sort_idx = np.argsort(dist_to_j)
                        
                        sorted_cust_i = cust_i_for_j[sort_idx]
                        sorted_req = req_for_j[sort_idx]
                        sorted_alloc = allocated[sort_idx]
                        
                        for idx_c in range(len(sorted_cust_i)):
                            if leftover == 0:
                                break
                            c_id = sorted_cust_i[idx_c]
                            if sorted_req[idx_c] > sorted_alloc[idx_c] and local_D[c_id] > 0:
                                F[c_id, abs_j] += 1
                                local_D[c_id] -= 1
                                leftover -= 1
                                
        return F

    def fitness(self, x):
        F = self.route(x)
        total_E, _, _ = E(x, F, self.C, self.P, self.R)
        total_O, _, _ = O(F, self.D, self.L, self.config['alpha'], self.config['beta'])
        
        return self.config['lambda'] * total_E - self.config.get('mu', 1.0) * total_O

    def get_details(self, x):
        F = self.route(x)
        total_E, rev, cost = E(x, F, self.C, self.P, self.R)
        total_O, unmet, dist = O(F, self.D, self.L, self.config['alpha'], self.config['beta'])
        
        return {
            'avg_E': total_E, 'avg_rev': rev, 'avg_cost': cost,
            'avg_O': total_O, 'avg_unmet': unmet, 'avg_dist': dist
        }

###################
#### Experiment

class Experiment:
    def __init__(self, data, experiment_name, config):
        self.N, self.B, self.C, self.P, self.L, self.R, self.Z, self.D = data
        self.experiment_name = experiment_name
        self.config = config
            
        self.model = self.config['model_builder'](
            self.N, self.B, self.C, self.P, self.R, self.L, self.Z, self.D, self.config
        )
        self.generation_history = []

    @cache
    def _fitness_handler_helper(self, solution):
        return self.model.fitness(solution)

    def fitness_handler(self, ga_instance, solution, solution_idx):
        return self._fitness_handler_helper(tuple(solution.tolist()))

    def log_handler(self, ga_instance):
        best_sol, best_fit, _ = ga_instance.best_solution()
        metrics = self.model.get_details(best_sol)
        
        self.generation_history.append({
            "generation": ga_instance.generations_completed,
            "best_average_fitness": float(best_fit),
            "avg_E_profit": float(metrics['avg_E']),
            "avg_E_revenue": float(metrics['avg_rev']),
            "avg_E_cost": float(metrics['avg_cost']),
            "avg_O_loss": float(metrics['avg_O']),
            "avg_O_unmet_penalty": float(metrics['avg_unmet']),
            "avg_O_distance_penalty": float(metrics['avg_dist']),
            "x": best_sol.tolist()
        })
        
        print(f"[{self.experiment_name}] Gen {ga_instance.generations_completed:02d} | Stations: {int(np.sum(best_sol)):03d} | Fit: {best_fit:,.2f} | "
              f"E: {metrics['avg_E']:,.2f} (Rev: {metrics['avg_rev']:,.2f}, Cost: {metrics['avg_cost']:,.2f}) | "
              f"O: {metrics['avg_O']:,.2f} (Unmet: {metrics['avg_unmet']:,.2f}, Dist: {metrics['avg_dist']:,.2f})")

    def run(self):
        print(f"--- Starting Experiment ({self.experiment_name}) ---")
        
        ga_instance = pygad.GA(
            num_generations=self.config['num_generations'],
            num_parents_mating=self.config['num_parents_mating'],
            fitness_func=self.fitness_handler,
            sol_per_pop=self.config['sol_per_pop'], 
            num_genes=self.N,
            stop_criteria=self.config['stop_criteria'],
            on_generation=self.log_handler,
            random_seed=self.config['random_seed'],
            gene_type=int,
            gene_space=[0, 1],
            parent_selection_type=self.config['parent_selection_type'],
            K_tournament=self.config['K_tournament'],
            crossover_type=self.config['crossover_type'],
            mutation_type=self.config['mutation_type'],
            mutation_probability=self.config['mutation_probability'],
            keep_elitism=self.config['keep_elitism']
        )

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

        full_model_name = f"{self.model.name}_{self.experiment_name}"
        
        save_optimization_results(
            best_x, 
            best_fitness, 
            self.generation_history, 
            self.config, 
            input_path.name,
            start_time_str,
            end_time_str,
            run_time,
            full_model_name
        )
        return best_x, best_fitness


if __name__ == "__main__": 
    data_tuple = read_input()
    
    exp1 = Experiment(data=data_tuple, experiment_name="Customer Routing", config= {
        'alpha': 10.0,
        'beta': 0.0005,
        'lambda': 1.0,
        'mu': 1.0,           
        'K': 10,             
        'w1': 0.95,           
        'w2': 0.05,           
        'gamma': 1.0,        
        'epsilon': 1.0,      
        'model_builder': CustomerRouting, 
        'num_generations': 50,
        'sol_per_pop': 100,  
        'num_parents_mating': 10,
        'num_shuffles': 3,
        'random_seed': 42,
        'stop_criteria': ['saturate_50'],
        'parent_selection_type': 'tournament',
        'K_tournament': 3,
        'crossover_type': 'uniform', 
        'mutation_type': 'adaptive',
        'mutation_probability': [0.35, 0.05],
        'keep_elitism': 5
    })
    exp1.run()
    
    exp2 = Experiment(data=data_tuple, experiment_name="Behavioral Routing", config={
        'alpha': 10.0,
        'beta': 0.0005,
        'lambda': 1.0,
        'mu': 1.0,           
        'K': 10,             
        'w1': 0.8,           
        'w2': 0.2,           
        'gamma': 1.0,        
        'epsilon': 1.0,      
        'model_builder': BehavioralRouting, 
        'num_generations': 200,
        'sol_per_pop': 100,  
        'num_parents_mating': 10,
        'num_shuffles': 3,
        'random_seed': 42,
        'stop_criteria': ['saturate_50'],
        'parent_selection_type': 'tournament',
        'K_tournament': 3,
        'crossover_type': custom_intersection_crossover, 
        'mutation_type': 'adaptive',
        'mutation_probability': [0.35, 0.05],
        'keep_elitism': 5
    })
    exp2.run()