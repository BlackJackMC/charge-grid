import pygad
import numpy as np
from pathlib import Path
from datetime import datetime
from functools import cache

from models.cluster import ClusterRouting
from models.customer import CustomerRouting
from models.station import StationRouting
from models.behavioral import BehavioralRouting
from models.milp_routing import MILPRoutingORTools
from utils import read_input, save_optimization_results, custom_intersection_crossover, smart_add_drop_mutation, on_generation_earthquake

input_folder = Path('..')
output_folder = Path('./output')
input_path = input_folder / 'input_q1.txt'

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
            keep_elitism=self.config['keep_elitism'],
        )

        ga_instance.D = np.array(self.D)
        ga_instance.L = np.array(self.L)

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
            output_folder,
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
    data_tuple = read_input(input_path)
    
    exp1 = Experiment(data=data_tuple, experiment_name="Customer Routing", config= {
        'alpha': 10.0,
        'beta': 0.0005,
        'lambda': 1.0,
        'model_builder': CustomerRouting, 
        'num_generations': 200,
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
    
    exp2 = Experiment(data=data_tuple, experiment_name="Behavioral Routing", config={
        'alpha': 10.0,
        'beta': 0.0005,
        'lambda': 1.0,
        'mu': 1.0,           
        'K': 10,             
        'w1': 0.95,           
        'w2': 0.05,           
        'gamma': 1.0,        
        'epsilon': 1.0,      
        'model_builder': BehavioralRouting, 
        'num_generations': 5000,
        'sol_per_pop': 300,  
        'num_parents_mating': 20,
        'num_shuffles': 1,
        'random_seed': 32,
        'stop_criteria': ['saturate_1000'],
        'parent_selection_type': 'tournament',
        'K_tournament': 10,
        'crossover_type': custom_intersection_crossover, 
        'mutation_type': smart_add_drop_mutation,
        'mutation_probability': None,
        'keep_elitism': 5
    })
    
    exp3 = Experiment(data=data_tuple, experiment_name="Customer Routing", config= {
        'alpha': 100.0,
        'beta': 0.0005,
        'lambda': 1.0,
        'model_builder': CustomerRouting, 
        'num_generations': 200,
        'sol_per_pop': 100,  
        'num_parents_mating': 10,
        'num_shuffles': 10,
        'random_seed': 42,
        'stop_criteria': ['saturate_50'],
        'parent_selection_type': 'tournament',
        'K_tournament': 3,
        'crossover_type': 'uniform',
        'mutation_type': 'adaptive',
        'mutation_probability': [0.35, 0.05],
        'keep_elitism': 5
    })
    
    exp4 = Experiment(data=data_tuple, experiment_name="Station Routing", config= {
        'alpha': 100.0,
        'beta': 0.0005,
        'lambda': 1.0,
        'model_builder': StationRouting, 
        'num_generations': 200,
        'sol_per_pop': 100,  
        'num_parents_mating': 10,
        'num_shuffles': 10,
        'random_seed': 42,
        'stop_criteria': ['saturate_50'],
        'parent_selection_type': 'tournament',
        'K_tournament': 3,
        'crossover_type': 'uniform',
        'mutation_type': 'adaptive',
        'mutation_probability': [0.35, 0.05],
        'keep_elitism': 5
    })

    exp5 = Experiment(data=data_tuple, experiment_name="Station Cluster Routing", config= {
        'alpha': 100.0,
        'beta': 0.0005,
        'lambda': 1.0,
        'model_builder': ClusterRouting,
        'num_clusters': 40,
        'num_generations': 200,
        'sol_per_pop': 100,  
        'num_parents_mating': 10,
        'num_shuffles': 10,
        'random_seed': 42,
        'stop_criteria': ['saturate_50'],
        'parent_selection_type': 'tournament',
        'K_tournament': 3,
        'crossover_type': 'uniform',
        'mutation_type': 'adaptive',
        'mutation_probability': [0.35, 0.05],
        'keep_elitism': 5
    })

    exp6 = Experiment(data=data_tuple, experiment_name="Station Cluster Routing", config= {
        'alpha': 10.0,
        'beta': 0.0005,
        'lambda': 1.0,
        'model_builder': ClusterRouting,
        'num_clusters': 40,
        'num_generations': 5000,
        'sol_per_pop': 300,  
        'num_parents_mating': 20,
        'num_shuffles': 1,
        'random_seed': 4,
        'stop_criteria': ['saturate_1000'],
        'parent_selection_type': 'tournament',
        'K_tournament': 10,
        'crossover_type': custom_intersection_crossover,
        'mutation_type': smart_add_drop_mutation,
#        'mutation_probability': [0.35, 0.05],
        'keep_elitism': 5
    })

    exp_milp = Experiment( data=data_tuple, experiment_name="MILP Only OR-Tools", config={
        'alpha': 10.0,
        'beta': 0.0005,
        'lambda': 1.0,

        'model_builder': MILPRoutingORTools,

        # ⚠️ MUST be small (MILP is expensive)
        'num_generations': 500,
        'sol_per_pop': 40,
        'num_parents_mating': 20,
        'random_seed': 67,
        'stop_criteria': ['saturate_200'],
        'parent_selection_type': 'tournament',
        'K_tournament': 4,
        'crossover_type': 'uniform',
        'mutation_type': smart_add_drop_mutation,
#        'mutation_probability': [0.3, 0.05],
        'keep_elitism': 3,
        # ⏱️ VERY IMPORTANT
        'milp_time_limit': 5000  # milliseconds (2 seconds)
    }
)
    exp2.run()