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
    offspring = []
    idx = 0
    target_size = offspring_size if isinstance(offspring_size, tuple) else offspring_size
    num_offspring_needed = offspring_size[0]
    while len(offspring) < num_offspring_needed:
        parent1 = parents[idx % parents.shape[0], :]
        parent2 = parents[(idx + 1) % parents.shape[0], :]
        agree_mask = (parent1 == parent2)
        take_from_p1 = np.random.rand(*parent1.shape) < 0.80
        
        child = np.where(agree_mask, parent1, np.where(take_from_p1, parent1, parent2))
        offspring.append(child)
        idx += 1
        
    return np.array(offspring)



def smart_add_drop_mutation(offspring, ga_instance):
    # Retrieve the demand data and distance matrix attached to the GA instance
    D = ga_instance.D
    L = ga_instance.L 
    
    # FIX 1: Use offspring.shape to get the integer number of rows
    for i in range(offspring.shape[0]):
        chromosome = offspring[i]
        
        # 50/50 chance to either ADD a station or DROP a station
        if np.random.rand() < 0.5:
            # --- SMART ADD (Demand-Weighted) ---
            # FIX 2: Add  to extract the 1D array of indices
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
            # --- SMART DROP (Coverage-Aware / Redundancy-Aware) ---
            # FIX 3: Add  to extract the 1D array of indices
            active_spots = np.where(chromosome == 1)[0]
            
            # We need at least 2 active stations to calculate redundancy between them
            if len(active_spots) > 1:
                # 1. Create a sub-matrix of distances between ONLY the currently active stations
                active_distances = L[np.ix_(active_spots, active_spots)].copy()
                
                # 2. Ignore the distance from a station to itself by setting the diagonal to infinity
                np.fill_diagonal(active_distances, np.inf)
                
                # 3. For each active station, find the distance to its closest active neighbor
                min_dists = np.min(active_distances, axis=1)
                
                # 4. Stations that are very close together have a SMALL min_dist (high redundancy).
                # We invert this so that small distances become HIGH probabilities for being dropped.
                redundancy_scores = 1.0 / (min_dists + 1e-9)
                
                # 5. Normalize into a probability distribution
                drop_probabilities = redundancy_scores / np.sum(redundancy_scores)
                
                # 6. Pick the station to drop, heavily biased towards redundant stations
                chosen_drop = np.random.choice(active_spots, p=drop_probabilities)
                chromosome[chosen_drop] = 0
                
            elif len(active_spots) == 1:
                # Fallback if only 1 station is active
                chosen_drop = active_spots
                chromosome[chosen_drop] = 0
                
    return offspring


def on_generation_earthquake(self, ga_instance):
    # 1. Initialize tracking variables on the first generation
    if not hasattr(ga_instance, 'last_fitness'):
        ga_instance.last_fitness = ga_instance.best_solution()[1]
        ga_instance.stagnation_counter = 0

    # 2. Check the current best fitness
    current_fitness = ga_instance.best_solution()[1]

    # 3. Increment counter if stuck, or reset if we improved
    if current_fitness == ga_instance.last_fitness:
        ga_instance.stagnation_counter += 1
    else:
        ga_instance.last_fitness = current_fitness
        ga_instance.stagnation_counter = 0

    # 4. TRIGGER THE EARTHQUAKE (e.g., after 15 generations of stagnation)
    if ga_instance.stagnation_counter >= 300:
        print(f"\n[!] Stagnation detected at Gen {ga_instance.generations_completed}. Triggering Earthquake...")

        # Find and save the absolute best chromosome (The Elite)
        best_solution, _, best_idx = ga_instance.best_solution()
        elite_chromosome = best_solution.copy()

        # Generate an entirely new, completely random population
        num_chromosomes, chromosome_length = ga_instance.population.shape
        # Adjust the probability here if you want sparse arrays (e.g., mostly 0s)
        random_population = np.random.choice([0,1], size=(num_chromosomes, chromosome_length), p=[0.9, 0.1])

        # Wipe out the old, converged population
        ga_instance.population = random_population

        # Re-inject the Elite chromosome into the new population so it isn't lost
        ga_instance.population[0] = elite_chromosome

        # Reset the stagnation counter so the new population has time to evolve
        ga_instance.stagnation_counter = 0
        self.on_generation_earthquake(ga_instance)