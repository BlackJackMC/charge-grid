import pygad
import random
import json
import folium
import pandas as pd
import os
from pathlib import Path
from datetime import datetime

input_folder = Path('..')
output_folder = Path('./output')
input_path = input_folder / 'input_q1.txt'

#### I/O functions

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

def save_optimization_results(best_x, best_fitness, generation_history, config, input_filename):
    clean_config = {k: (v.__name__ if callable(v) else v) for k, v in config.items()}

    output_data = {
        "metadata": {
            "input_file": input_filename,
            "configuration": clean_config
        },
        "best_solution": {
            "x": best_x,
            "fitness_score": best_fitness
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

#### Core Logic

def route(x, demand_order=None) -> list[list[int]]:
    station_battery = {i: B for i in range(N) if x[i] == 1}

    # Much faster: simply filter the pre-sorted list based on active stations
    nearest_stations = {
        i: [j for j in precomputed_nearest[i] if x[j] == 1]
        for i in range(N)
    }

    F = [[0 for _ in range(N)] for _ in range(N)]

    if demand_order is None:
        demand_order = list(range(len(D)))
        random.shuffle(demand_order)

    local_D = list(D)

    for i in demand_order:
        for j in nearest_stations[i]:
            if station_battery[j] > 0:
                F[i][j] = min(local_D[i], station_battery[j])
                local_D[i] -= F[i][j]
                station_battery[j] -= F[i][j]

    return F

def E(x, F):
    profit_val = 0
    for i in range(N):
        for j in range(N):
            profit_val += x[j] * F[i][j] * P

    for j in range(N):
        profit_val -= x[j] * (C + R[j])

    return profit_val

def O(F, alpha: float = 1, beta: float = 1):
    dissatisfaction = 0
    for i in range(N):
        dissatisfaction += alpha * (D[i] - sum(F[i]))
        for j in range(N):
            dissatisfaction += beta * F[i][j] * L[i][j]

    return dissatisfaction

def fitness(x):
    fitness_vals = []
    
    for order in evaluation_orders:
        F = config['behavior_model'](x, demand_order=order)
        fit = config['lambda'] * E(x, F) - O(F, config['alpha'], config['beta'])
        fitness_vals.append(fit)
        
    return sum(fitness_vals) / config['num_shuffles']

###################

#### GA callback & execution functions

def fitness_handler(ga_instance, solution, solution_idx):
    x = [int(val) for val in solution]
    return fitness(x)

def log_handler(ga_instance):
    best_sol, best_fit, _ = ga_instance.best_solution()
    current_x = [int(val) for val in best_sol]
    
    e_vals = []
    o_vals = []
    
    for order in evaluation_orders:
        F = config['behavior_model'](current_x, demand_order=order)
        e_vals.append(E(current_x, F))
        o_vals.append(O(F, config['alpha'], config['beta']))
        
    avg_E = sum(e_vals) / config['num_shuffles']
    avg_O = sum(o_vals) / config['num_shuffles']
    
    generation_history.append({
        "generation": ga_instance.generations_completed,
        "best_average_fitness": best_fit,
        "avg_E_profit": avg_E,
        "avg_O_loss": avg_O,
        "x": current_x
    })
    
    print(f"Generation {ga_instance.generations_completed:02d} | Fitness: {best_fit:,.2f} | Profit (E): {avg_E:,.2f} | Dissatisfaction (O): {avg_O:,.2f}")

def run_optimization(ga_instance, model_name):
    print(f"--- Starting Optimization ({model_name}) ---")
    ga_instance.run()

    best_x, best_fitness, _ = ga_instance.best_solution()
    best_x = [int(val) for val in best_x]

    print("\n--- Optimization Complete ---")
    print(f"Optimal Station Locations (x): {best_x}")
    print(f"Optimal Fitness Found: {best_fitness:,.2f}")
    
    return best_x, best_fitness

###################

if __name__ == "__main__": 
    problem_data = read_input()
    N, B, C, P, L, R, Z, D = problem_data


    precomputed_nearest = []
    for i in range(N):
        sorted_indices = sorted(range(N), key=lambda j: L[i][j])
        valid_indices = [j for j in sorted_indices if L[i][j] <= Z[i]]
        precomputed_nearest.append(valid_indices)
    
    config = {
        'alpha': 10.0,
        'beta': 0.0005,
        'lambda': 1.0,
        'behavior_model': route,
        'num_generations': 50,
        'sol_per_pop': 20,
        'num_parents_mating': 10,
        'mutation_percent_genes': 10,
        'num_shuffles': 5,
        # 'random_seed': int(datetime.now().timestamp()),
        'random_seed': 42,
        'stop_criteria': ['saturate_10'],
    }
    
    rng = random.Random(config['random_seed'])

    evaluation_orders = []
    for _ in range(config['num_shuffles']):
        order = list(range(N))
        rng.shuffle(order)
        evaluation_orders.append(order)

    generation_history = []

    ga_instance = pygad.GA(
        num_generations=config['num_generations'],
        num_parents_mating=config['num_parents_mating'],
        fitness_func=fitness_handler,
        sol_per_pop=config['sol_per_pop'],
        num_genes=N,
        gene_type=int,
        gene_space=[0, 1],
        stop_criteria=config['stop_criteria'],
        mutation_percent_genes=config['mutation_percent_genes'],
        on_generation=log_handler,
        random_seed=config['random_seed']
    )

    best_x, best_fitness = run_optimization(ga_instance, config['behavior_model'].__name__)

    save_optimization_results(
        best_x, 
        best_fitness, 
        generation_history, 
        config, 
        input_path.name
    )

csv_metadata_path = input_folder / 'data_hcm.csv'
try:
    df_meta = pd.read_csv(csv_metadata_path, nrows=540)
except Exception as e:
    df_meta = None

if df_meta is not None:
    map_folder = output_folder / 'html_maps'
    map_folder.mkdir(parents=True, exist_ok=True)
    map_rng = random.Random(42)
    five_orders = []
    for _ in range(5):
        order = list(range(N))
        map_rng.shuffle(order)
        five_orders.append(order)
    for idx, current_order in enumerate(five_orders):
        print(f"Đang xử lý Bản đồ {idx + 1}/5...")
        F_matrix = config['behavior_model'](best_x, demand_order=current_order)
        center_lat = df_meta['lat'].mean()
        center_lon = df_meta['lon'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='CartoDB positron')
        for i in range(N):
            for j in range(N):
                if F_matrix[i][j] > 0:
                    if i < len(df_meta) and j < len(df_meta):
                        loc_i = [df_meta.iloc[i]['lat'], df_meta.iloc[i]['lon']]
                        loc_j = [df_meta.iloc[j]['lat'], df_meta.iloc[j]['lon']]
                        line_weight = max(1.0, F_matrix[i][j] * 0.2)
                        folium.PolyLine(
                            locations=[loc_i, loc_j],
                            color='#3186cc',
                            weight=line_weight,
                            opacity=0.5,
                            dash_array='5, 5',
                            tooltip=f"Khách {df_meta.iloc[i]['name']} -> Trạm {df_meta.iloc[j]['name']}: {F_matrix[i][j]} pin"
                        ).add_to(m)
        for i in range(N):
            if i < len(df_meta):
                loc = [df_meta.iloc[i]['lat'], df_meta.iloc[i]['lon']]
                name = df_meta.iloc[i]['name']
                if 'district' in df_meta.columns:
                    district = df_meta.iloc[i]['district']
                else:
                    district = "N/A"
                
                if best_x[i] == 1:
                    pin_ban_duoc = sum(F_matrix[k][i] for k in range(N))
                    folium.Marker(
                        location=loc,
                        tooltip=f"TRẠM SẠC: {name}",
                        popup=f"<b>Trạm sạc:</b> {name}<br><b>Quận:</b> {district}<br><b>Số pin đã phục vụ:</b> {pin_ban_duoc}",
                        icon=folium.Icon(color='green', icon='bolt', prefix='fa')
                    ).add_to(m)
                else:
                    nhu_cau_duoc_dap_ung = sum(F_matrix[i][k] for k in range(N))
                    if nhu_cau_duoc_dap_ung > 0:
                        folium.CircleMarker(
                            location=loc,
                            radius=3,
                            color='red',
                            fill=True,
                            fill_color='red',
                            fill_opacity=0.7,
                            tooltip=f"KHÁCH: {name}",
                            popup=f"<b>Khách hàng:</b> {name}<br><b>Quận:</b> {district}<br><b>Đã đổi:</b> {nhu_cau_duoc_dap_ung} pin"
                        ).add_to(m)

        map_filename = map_folder / f"Map_Result_Seed42_Shuffle_{idx + 1}.html"
        m.save(str(map_filename))
    
