from pathlib import Path

from charge_grid.experiment import Experiment

from charge_grid.models.cluster import ClusterRouting
from charge_grid.models.customer import CustomerRouting
from charge_grid.models.station import StationRouting
from charge_grid.models.alternating import AlternatingRouting

from charge_grid.utils import read_input, custom_intersection_crossover, adaptive_mutation, stagnation_aware_adaptive_mutation, noise_injected_adaptive_mutation, INPUT_DIR, OUTPUT_DIR

input_path = INPUT_DIR / 'input_hcm.txt'
data_tuple = read_input(input_path)

exp1 = Experiment(data=data_tuple, input_path=input_path, output_folder=OUTPUT_DIR, 
                  experiment_name="Customer Routing - soft constraint", config= {
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


exp3 = Experiment(data=data_tuple, input_path=input_path, output_folder=OUTPUT_DIR, 
                  experiment_name="Customer Routing", config= {
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

exp4 = Experiment(data=data_tuple, input_path=input_path, output_folder=OUTPUT_DIR, 
                  experiment_name="Station Routing", config= {
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

exp5 = Experiment(data=data_tuple, input_path=input_path, output_folder=OUTPUT_DIR, 
                  experiment_name="Station Cluster Routing", config= {
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

exp6 = Experiment(data=data_tuple, input_path=input_path, output_folder=OUTPUT_DIR, 
                  experiment_name="Station Cluster Routing - adaptive mutation", config= {
    'alpha': 10.0,
    'beta': 0.0005,
    'lambda': 1.0,
    'model_builder': ClusterRouting,
    'num_clusters': 40,
    'num_generations': 750,
    'sol_per_pop': 100,  
    'num_parents_mating': 10,
    'num_shuffles': 2,
    'random_seed': 42,
    'stop_criteria': ['saturate_50'],
    'parent_selection_type': 'tournament',
    'K_tournament': 5,
    'crossover_type': custom_intersection_crossover,
    'mutation_type': noise_injected_adaptive_mutation,
    'keep_elitism': 5
})

exp7 = Experiment(data=data_tuple, input_path=input_path, output_folder=OUTPUT_DIR, 
                  experiment_name="Stable Matching - adaptive mutation", config= {
    'alpha': 100.0,
    'beta': 0.0005,
    'lambda': 1.0,
    'model_builder': AlternatingRouting,
    'num_clusters': 40,
    'num_generations': 200,
    'sol_per_pop': 100,  
    'num_parents_mating': 10,
    'random_seed': 42,
    'stop_criteria': ['saturate_100'],
    'parent_selection_type': 'tournament',
    'K_tournament': 3,
    'crossover_type': 'uniform',
    'mutation_type': adaptive_mutation(data_tuple),
    'mutation_probability': None,
    'keep_elitism': 5
})

exp8 = Experiment(data=data_tuple, input_path=input_path, output_folder=OUTPUT_DIR, 
                  experiment_name="Stable Matching - large generation - adaptive mutation", config= {
    'alpha': 100.0,
    'beta': 0.0005,
    'lambda': 1.0,
    'model_builder': AlternatingRouting,
    'num_clusters': 40,
    'num_generations': 750,
    'sol_per_pop': 100,  
    'num_parents_mating': 10,
    'random_seed': 42,
    'stop_criteria': ['saturate_100'],
    'parent_selection_type': 'tournament',
    'K_tournament': 5,
    'crossover_type': 'uniform',
    'mutation_type': adaptive_mutation(data_tuple),
    'mutation_probability': None,
    'keep_elitism': 5
})

exp9 = Experiment(data=data_tuple, input_path=input_path, output_folder=OUTPUT_DIR, 
                  experiment_name="Stable Matching - large generation - stagnation aware adaptive mutation", config= {
    'alpha': 100.0,
    'beta': 0.0005,
    'lambda': 1.0,
    'model_builder': AlternatingRouting,
    'num_clusters': 40,
    'num_generations': 750,
    'sol_per_pop': 100,  
    'num_parents_mating': 10,
    'random_seed': 42,
    'stop_criteria': ['saturate_100'],
    'parent_selection_type': 'tournament',
    'K_tournament': 5,
    'crossover_type': 'uniform',
    'mutation_type': stagnation_aware_adaptive_mutation(data_tuple),
    'mutation_probability': None,
    'keep_elitism': 5
})

exp10 = Experiment(data=data_tuple, input_path=input_path, output_folder=OUTPUT_DIR,
                    experiment_name="Stable Matching - large generation - stagnation aware adaptive mutation - custom crossover intersection", config= {
    'alpha': 100.0,
    'beta': 0.0005,
    'lambda': 1.0,
    'model_builder': AlternatingRouting,
    'num_clusters': 40,
    'num_generations': 750,
    'sol_per_pop': 100,  
    'num_parents_mating': 10,
    'random_seed': 42,
    'stop_criteria': ['saturate_100'],
    'parent_selection_type': 'tournament',
    'K_tournament': 5,
    'crossover_type': custom_intersection_crossover(data_tuple),
    'mutation_type': noise_injected_adaptive_mutation(data_tuple),
    'mutation_probability': None,
    'keep_elitism': 5
})

def main():
    for crossover in ['uniform']:
        for mutation in [noise_injected_adaptive_mutation(data_tuple)]:
            exp_temp = Experiment(data=data_tuple, input_path=input_path, output_folder=OUTPUT_DIR, 
                              experiment_name="Stable Matching - large generation - all mutation - all crossover", config= {
            'alpha': 100.0,
            'beta': 0.0005,
            'lambda': 1.0,
            'model_builder': AlternatingRouting,
            'num_clusters': 40,
            'num_generations': 750,
            'sol_per_pop': 120,
            'num_parents_mating': 12,
            'stop_criteria': ['saturate_100'],
            'parent_selection_type': 'tournament',
            'random_seed': 42,
            'K_tournament': 5,
            'crossover_type': crossover,
            'mutation_type': mutation,
            'mutation_probability': None,
            'keep_elitism': 5
        })
            
        exp_temp.run()

    # for model in [CustomerRouting, BehavioralRouting, StationRouting, ClusterRouting, AlternatingRouting]:
    #     exp_temp = Experiment(data=data_tuple, input_path=input_path, output_folder=OUTPUT_DIR, 
    #                           experiment_name="All model - large generation - adaptive mutation", config= {
    #         'alpha': 100.0,
    #         'beta': 0.0005,
    #         'lambda': 1.0,
    #         'mu': 1.0,
    #         'K': 10,
    #         'w1': 0.95,
    #         'w2': 0.05,
    #         'gamma': 1.0,
    #         'epsilon': 1.0,
    #         'model_builder': model,
    #         'num_clusters': 40,
    #         'num_generations': 750,
    #         'sol_per_pop': 100,
    #         'num_parents_mating': 10,
    #         'random_seed': 42,
    #         'num_shuffles': 5,
    #         'stop_criteria': ['saturate_100'],
    #         'parent_selection_type': 'tournament',
    #         'K_tournament': 5,
    #         'crossover_type': 'uniform',
    #         'mutation_type': noise_injected_adaptive_mutation(data_tuple),
    #         'mutation_probability': None,
    #         'keep_elitism': 5
    #     })

    pass

if __name__ == "__main__": 
    main()
