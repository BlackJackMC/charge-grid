import pygad
import random
import json
import folium
import pandas as pd
import os
import requests
import numpy as np
import math
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

def route(x, station_order=None) -> list[list[int]]:
    """
    Iterative routing logic allocating batteries in chunks K, 
    evaluating distance, station congestion, and battery depletion.
    """
    K = config['K']
    w1 = config['w1']
    w2 = config['w2']
    gamma = config['gamma']
    epsilon = config['epsilon']
    F = [[0 for _ in range(N)] for _ in range(N)]
    station_battery = {j: B for j in range(N) if x[j] == 1}
    local_D = list(D)
    
    if sum(local_D) == 0 or not station_battery:
        return F

    V = {j: 0 for j in station_battery}
    for j in station_battery:
        for k in range(N):
            if L[k][j] <= Z[k]:
                V[j] += D[k]

    while True: # Thay vòng lặp for bằng vòng lặp while vô tận
        if sum(local_D) == 0:
            break            
        requested_amount = {j: 0 for j in station_battery}
        target_facility = {i: -1 for i in range(N)}
        delta_req = {i: 0 for i in range(N)}
        
        for i in range(N):
            if local_D[i] > 0:
                min_loss = float('inf')
                best_j = -1
                for j in station_battery:
                    if station_battery[j] > 0 and L[i][j] <= Z[i]:
                        norm_dist = L[i][j] / Z[i] if Z[i] > 0 else 0
                        congestion = V[j] / B if B > 0 else 0
                        depletion = B / (station_battery[j] + epsilon)
                        R_ij = norm_dist * congestion * depletion
                        current_loss = w1 * norm_dist + w2 * math.log(1 + R_ij)                
                        if current_loss < min_loss:
                            min_loss = current_loss
                            best_j = j
                
                if best_j!= -1:
                    target_facility[i] = best_j
                    # SỬA LẠI CÁCH TÍNH CHUNK BẰNG MATH.CEIL:
                    chunk = max(1, math.ceil(D[i] / K)) 
                    delta_req[i] = min(local_D[i], chunk)
                    requested_amount[best_j] += delta_req[i]
        if sum(requested_amount.values()) == 0:
            break
            
        for j in station_battery:
            if requested_amount[j] == 0:
                continue    
            targeting_customers = [i for i in range(N) if target_facility[i] == j]
            
            if requested_amount[j] <= station_battery[j]:
                for i in targeting_customers:
                    F[i][j] += delta_req[i]
                    local_D[i] -= delta_req[i]
                    station_battery[j] -= delta_req[i]
            else:
                ratio = station_battery[j] / requested_amount[j]
                leftover_batteries = station_battery[j]
                for i in targeting_customers:
                    allocated = int(delta_req[i] * ratio) 
                    F[i][j] += allocated
                    local_D[i] -= allocated
                    leftover_batteries -= allocated
                    
                if leftover_batteries > 0:
                    targeting_customers.sort(key=lambda c: L[c][j])
                    for i in targeting_customers:
                        if leftover_batteries == 0:
                            break
                        allocated_so_far = int(delta_req[i] * ratio)
                        if delta_req[i] > allocated_so_far and local_D[i] > 0:
                            F[i][j] += 1
                            local_D[i] -= 1
                            leftover_batteries -= 1
                            
                station_battery[j] = 0

    return F

def E(x, F):
    revenue = 0
    cost = 0

    for i in range(N):
        for j in range(N):
            revenue += x[j] * F[i][j] * P

    for j in range(N):
        cost += x[j] * (C + R[j])

    total_profit = revenue - cost
    return total_profit, revenue, cost

def O(F, alpha: float = 1, beta: float = 1):
    unmet_penalty = 0
    distance_penalty = 0

    for i in range(N):
        unmet_penalty += alpha * (D[i] - sum(F[i]))
        for j in range(N):
            distance_penalty += beta * F[i][j] * L[i][j]

    total_dissatisfaction = unmet_penalty + distance_penalty
    return total_dissatisfaction, unmet_penalty, distance_penalty

def fitness(x):
    fitness_vals = []
    
    for order in evaluation_orders:
        F = config['behavior_model'](x, station_order=order)
        total_E, _, _ = E(x, F)
        total_O, _, _ = O(F, config['alpha'], config['beta'])
        # Updated to include mu penalty scaling
        fit = config['lambda'] * total_E - config['mu'] * total_O
        fitness_vals.append(fit)
        
    return sum(fitness_vals) / config['num_shuffles']

###################

#### GA callback & execution functions

def fitness_handler(ga_instance, solution, solution_idx):
    x = [int(val) for val in solution]
    return fitness(x)

def custom_intersection_crossover(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    while len(offspring) < offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        agree_mask = (parent1 == parent2)
        random_genes = np.random.randint(0, 2, size=parent1.shape)
        child = np.where(agree_mask, parent1, random_genes)

        offspring.append(child)
        idx += 1
        
    return np.array(offspring)

def custom_smart_mutation(offspring, ga_instance):
    for chromosome_idx in range(offspring.shape[0]):
        num_mutations = max(1, int((ga_instance.mutation_percent_genes / 100) * offspring.shape[1]))

        for _ in range(num_mutations):
            if np.random.rand() < 0.80: 
                ones_indices = np.where(offspring[chromosome_idx] == 1)[0]
                zeros_indices = np.where(offspring[chromosome_idx] == 0)[0]
                
                if len(ones_indices) > 0 and len(zeros_indices) > 0:
                    idx1 = np.random.choice(ones_indices)
                    idx2 = np.random.choice(zeros_indices)
                    offspring[chromosome_idx][idx1] = 0
                    offspring[chromosome_idx][idx2] = 1
            else:
                random_idx = np.random.randint(0, offspring.shape[1])
                offspring[chromosome_idx][random_idx] = 1 - offspring[chromosome_idx][random_idx]

    return offspring

def log_handler(ga_instance):
    best_sol, best_fit, _ = ga_instance.best_solution()
    current_x = [int(val) for val in best_sol]
    
    e_vals, rev_vals, cost_vals = [], [], []
    o_vals, unmet_vals, dist_vals = [], [], []
    
    for order in evaluation_orders:
        F = config['behavior_model'](current_x, station_order=order)
        total_E, rev, cost = E(current_x, F)
        total_O, unmet, dist = O(F, config['alpha'], config['beta'])
        
        e_vals.append(total_E)
        rev_vals.append(rev)
        cost_vals.append(cost)
        
        o_vals.append(total_O)
        unmet_vals.append(unmet)
        dist_vals.append(dist)
        
    avg_E = sum(e_vals) / config['num_shuffles']
    avg_rev = sum(rev_vals) / config['num_shuffles']
    avg_cost = sum(cost_vals) / config['num_shuffles']
    
    avg_O = sum(o_vals) / config['num_shuffles']
    avg_unmet = sum(unmet_vals) / config['num_shuffles']
    avg_dist = sum(dist_vals) / config['num_shuffles']
    
    generation_history.append({
        "generation": ga_instance.generations_completed,
        "best_average_fitness": best_fit,
        "avg_E_profit": avg_E,
        "avg_E_revenue": avg_rev,
        "avg_E_cost": avg_cost,
        "avg_O_loss": avg_O,
        "avg_O_unmet_penalty": avg_unmet,
        "avg_O_distance_penalty": avg_dist,
        "x": current_x
    })
    
    print(f"Gen {ga_instance.generations_completed:02d} | Stations: {sum(current_x):03d} | Fit: {best_fit:,.2f} | "
          f"E: {avg_E:,.2f} (Rev: {avg_rev:,.2f}, Cost: {avg_cost:,.2f}) | "
          f"O: {avg_O:,.2f} (Unmet: {avg_unmet:,.2f}, Dist: {avg_dist:,.2f})")

def run_optimization(ga_instance, model_name):
    print(f"--- Starting Optimization ({model_name}) ---")
    ga_instance.run()

    best_x, best_fitness, _ = ga_instance.best_solution()
    best_x = [int(val) for val in best_x]

    print("\n--- Optimization Complete ---")
    print(f"Optimal Station Locations (x): {best_x}")
    print(f"Optimal Fitness Found: {best_fitness:,.2f}")
    
    return best_x, best_fitness


###############
# Build map 
def build_map_data(best_x, F_matrix, df_meta, N, D, L, config):
    map_data = {'stations': {}, 'customers': {}, 'routes': []}
    
    for i in range(N):
        if D[i] > 0 and i < len(df_meta):
            demand_met = sum(F_matrix[i])
            map_data['customers'][i] = {
                'id': i, 'name': df_meta.iloc[i]['name'],
                'lat': df_meta.iloc[i]['lat'], 'lon': df_meta.iloc[i]['lon'],
                'total_demand': int(D[i]), 'demand_met': int(demand_met),
                'unmet_demand': int(D[i] - demand_met)
            }

    for j in range(N):
        if best_x[j] == 1 and j < len(df_meta):
            served_customers = []
            total_batt = 0
            station_dissat = 0
            
            for i in range(N):
                if F_matrix[i][j] > 0 and i < len(df_meta):
                    served_customers.append({'id': i, 'name': df_meta.iloc[i]['name'], 'batt': int(F_matrix[i][j])})
                    total_batt += F_matrix[i][j]
                    station_dissat += config['beta'] * F_matrix[i][j] * L[i][j]
            
            if total_batt > 0:
                map_data['stations'][j] = {
                    'id': j, 'name': df_meta.iloc[j]['name'],
                    'lat': df_meta.iloc[j]['lat'], 'lon': df_meta.iloc[j]['lon'],
                    'total_batt': int(total_batt), 'dissatisfaction': float(station_dissat),
                    'customers': served_customers
                }
    return map_data

def append_osrm_routes(map_data, F_matrix, df_meta, N, session):
    print("  > Lấy đường đi giao thông thực tế từ OSRM...")
    for i in range(N):
        for j in range(N):
            if F_matrix[i][j] > 0 and i < len(df_meta) and j < len(df_meta):
                lat_i, lon_i = df_meta.iloc[i]['lat'], df_meta.iloc[i]['lon']
                lat_j, lon_j = df_meta.iloc[j]['lat'], df_meta.iloc[j]['lon']
                
                route_coords = [[lat_i, lon_i], [lat_j, lon_j]] 
                try:
                    url = f"http://router.project-osrm.org/route/v1/driving/{lon_i},{lat_i};{lon_j},{lat_j}?overview=full&geometries=geojson"
                    res = session.get(url, timeout=2).json()
                    if res.get('code') == 'Ok':
                        route_coords = [[c[1], c[0]] for c in res['routes'][0]['geometry']['coordinates']]
                except Exception:
                    pass 

                map_data['routes'].append({
                    'cust_id': i, 'stat_id': j,
                    'coords': route_coords, 'batt': int(F_matrix[i][j])
                })
    return map_data

def generate_interactive_maps(best_x, df_meta, N, D, L, config):
    map_folder = output_folder / 'html_maps'
    map_folder.mkdir(parents=True, exist_ok=True)
    
    map_rng = random.Random(42)
    session = requests.Session()
    
    # Tính số trạm đã đặt
    num_stations = sum(best_x)
    
    for idx in range(5):
        print(f"\n--- Đang xử lý Bản đồ {idx + 1}/5 ---")
        
        current_order = list(range(N))
        map_rng.shuffle(current_order)
        F_matrix = config['behavior_model'](best_x, station_order=current_order)
        
        # Lấy Profit (E) và Dissatisfaction (O)
        total_profit_val, _, _ = E(best_x, F_matrix)
        total_dissat_val, _, _ = O(F_matrix, config['alpha'], config['beta'])
        
        map_data = build_map_data(best_x, F_matrix, df_meta, N, D, L, config)
        map_data = append_osrm_routes(map_data, F_matrix, df_meta, N, session)

        center_lat, center_lon = df_meta['lat'].mean(), df_meta['lon'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='CartoDB positron')

        custom_html = f"""
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css"/>
        <link rel="stylesheet" href="assets/map_style.css"/>
        
        <div class="main-stats-box">
            <div class="text-profit"><i class="fas fa-chart-line"></i> Lợi nhuận (E): {total_profit_val:,.2f}</div>
            <div class="text-dissat"><i class="fas fa-frown"></i> Dissatisfaction (O): {total_dissat_val:,.2f}</div>
            <div class="text-station"><i class="fas fa-charging-station"></i> Trạm đã đặt: {num_stations} / {N} node</div>
        </div>
        
        <div id="info-panel"></div>

        <script>
            window.map_data = {json.dumps(map_data)};
        </script>
        <script src="assets/map_script.js"></script>
        """
        
        m.get_root().html.add_child(folium.Element(custom_html))

        map_filename = map_folder / f"Map_Result_Seed42_Shuffle_{idx + 1}.html"
        m.save(str(map_filename))
        print(f"  > Đã lưu map thành công: {map_filename}")
###################

if __name__ == "__main__": 
    problem_data = read_input()
    N, B, C, P, L, R, Z, D = problem_data
    
    config = {
        'alpha': 10.0,
        'beta': 0.0005,
        'lambda': 1.0,
        'mu': 1.0,           # Added from second script
        'K': 10,             # Added from second script
        'w1': 0.4,           # Added from second script
        'w2': 0.6,           # Added from second script
        'gamma': 1.0,        # Added from second script
        'epsilon': 1.0,      # Added from second script
        'behavior_model': route,
        'num_generations': 20,
        'sol_per_pop': 30,
        'num_parents_mating': 10,
        'mutation_percent_genes': 2, 
        'num_shuffles': 5,
        'random_seed': int(datetime.now().timestamp()),
        'stop_criteria': ['saturate_20'],
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
        crossover_type=custom_intersection_crossover,
        mutation_type=custom_smart_mutation,
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
        generate_interactive_maps(best_x, df_meta, N, D, L, config)