import numpy as np
from datetime import datetime
import json

#### I/O Functions

def read_input(input_path):
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

def save_optimization_results(output_folder, best_x, best_fitness, generation_history, config, input_filename, start_time, end_time, run_time, model_name):
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
    target_size = offspring_size if isinstance(offspring_size, tuple) else offspring_size
    offspring =[]
    
    L = ga_instance.L 
    num_nodes = parents.shape[1]
    K = 10
    
    idx = 0
    while len(offspring) < target_size:
        parent1 = parents[idx % parents.shape, :]
        parent2 = parents[(idx + 1) % parents.shape, :]
        
        medoids = np.random.choice(num_nodes, size=K, replace=False)
        
        cluster_assignments = np.argmin(L[:, medoids], axis=1)
        
        inherit_from_p1 = np.random.rand(K) < 0.5
        
        child = np.zeros(num_nodes, dtype=parent1.dtype)
        
        for node_idx in range(num_nodes):
            node_cluster = cluster_assignments[node_idx]
            
            if inherit_from_p1[node_cluster]:
                child[node_idx] = parent1[node_idx]
            else:
                child[node_idx] = parent2[node_idx]
        
        offspring.append(child)
        idx += 1
        
    return np.array(offspring)


def smart_add_drop_mutation(data_tuple):
    D = data_tuple[7]
    L = data_tuple[4]
    B = data_tuple[1]
    def _smart_add_drop_mutation(offspring, ga_instance):
        total_demand = np.sum(D)
        is_stagnant = hasattr(ga_instance, 'stagnation_counter') and ga_instance.stagnation_counter > 50
        
        for i in range(offspring.shape[0]):
            chromosome = offspring[i]
            
            active_spots = np.where(chromosome == 1)[0]
            num_active = len(active_spots)
            if is_stagnant and num_active > 5:
                num_to_drop = max(2, int(num_active * 0.20))
                
                active_distances = L[np.ix_(active_spots, active_spots)].copy()
                np.fill_diagonal(active_distances, np.inf)
                
                min_dists = np.min(active_distances, axis=1)
                redundancy_scores = 1.0 / (min_dists + 1e-9)
                drop_probabilities = redundancy_scores / np.sum(redundancy_scores)
                
                chosen_drops = np.random.choice(active_spots, size=num_to_drop, replace=False, p=drop_probabilities)
                chromosome[chosen_drops] = 0
                
            else:
                if num_active == 0:
                    add_prob = 1.0
                else:
                    network_capacity = num_active * B
                    utilization = total_demand / network_capacity
                    add_prob = np.clip(utilization, 0.10, 0.95)
                    
                if np.random.rand() < add_prob:
                    empty_spots = np.where(chromosome == 0)[0]
                    if len(empty_spots) > 0:
                        demand_at_empty = D[empty_spots]
                        sum_demand = np.sum(demand_at_empty)
                        
                        if sum_demand > 0:
                            probabilities = demand_at_empty / sum_demand
                        else:
                            probabilities = np.ones(len(empty_spots)) / len(empty_spots)
                            
                        chosen_spot = np.random.choice(empty_spots, p=probabilities)
                        chromosome[chosen_spot] = 1
                        
                else:
                    if num_active > 1:
                        active_distances = L[np.ix_(active_spots, active_spots)].copy()
                        np.fill_diagonal(active_distances, np.inf)
                        
                        min_dists = np.min(active_distances, axis=1)
                        redundancy_scores = 1.0 / (min_dists + 1e-9)
                        drop_probabilities = redundancy_scores / np.sum(redundancy_scores)
                        
                        chosen_drop = np.random.choice(active_spots, p=drop_probabilities)
                        chromosome[chosen_drop] = 0
                        
                    elif num_active == 1:
                        chromosome[active_spots] = 0
                        
        return offspring
    return _smart_add_drop_mutation


###################