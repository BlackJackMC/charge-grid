import pygad
import numpy as np
from datetime import datetime
from functools import cache

from charge_grid.utils import save_optimization_results


class Experiment:
    def __init__(self, data, experiment_name, input_path, output_folder, config):
        self.N, self.B, self.C, self.P, self.L, self.R, self.Z, self.D = data
        self.experiment_name = experiment_name
        self.config = config
        self.input_path = input_path
        self.output_folder = output_folder
            
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
        
        pop_fitnesses = ga_instance.last_generation_fitness
        avg_pop_fitness = sum(pop_fitnesses) / len(pop_fitnesses)
        best_pop_fitness = max(pop_fitnesses)
        
        metrics = self.model.get_details(best_sol)
        
        self.generation_history.append({
            "generation": ga_instance.generations_completed,
            "global_best_fitness": float(best_fit),
            "population_best_fitness": float(best_pop_fitness),
            "population_avg_fitness": float(avg_pop_fitness),
            "avg_E_profit": float(metrics['avg_E']),
            "avg_E_revenue": float(metrics['avg_rev']),
            "avg_E_cost": float(metrics['avg_cost']),
            "avg_O_loss": float(metrics['avg_O']),
            "avg_O_unmet_penalty": float(metrics['avg_unmet']),
            "avg_O_distance_penalty": float(metrics['avg_dist']),
            "x": best_sol.tolist()
        })
        
        print(f"[{self.experiment_name}] Gen {ga_instance.generations_completed:02d} | Stations: {int(np.sum(best_sol)):03d} | "
              f"Pop Avg: {avg_pop_fitness:,.2f} | Pop Best: {best_pop_fitness:,.2f} | Global Best: {best_fit:,.2f} | "
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
            self.output_folder,
            best_x, 
            best_fitness, 
            self.generation_history, 
            self.config, 
            self.input_path.name,
            start_time_str,
            end_time_str,
            run_time,
            full_model_name
        )
        return best_x, best_fitness
