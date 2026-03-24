import json
import pandas as pd
import folium
import random
from branca.element import Element
from pathlib import Path

from utils import read_input
from models.customer import CustomerRouting

input_folder = Path('..')
output_folder = Path('./output')
cache_folder = Path('./cache')
input_path = input_folder / 'input_q1.txt'
metadata_path = input_folder / 'data_q1.csv'

def choose_solution_file():
    if not output_folder.exists(): return None
    json_files = list(output_folder.glob('solution_*.json'))
    if not json_files: return None
    for idx, file_path in enumerate(json_files):
        print(f"[{idx}] {file_path.name}")
    try:
        choice = int(input("\nEnter the index: "))
        return json_files[choice] if 0 <= choice < len(json_files) else None
    except ValueError: return None

def generate_flow_map(best_x, N, B, D, F_list):
    try:
        df_d1 = pd.read_csv(metadata_path)
    except FileNotFoundError: return

    max_nodes = min(len(df_d1), N)
    m = folium.Map(location=[df_d1.head(max_nodes)['lat'].mean(), df_d1.head(max_nodes)['lon'].mean()], 
                   zoom_start=14, prefer_canvas=True, tiles='CartoDB positron')

    colors = ['#8000ff', '#e6194b', '#3cb44b', '#f58231', '#4363d8', '#911eb4', '#42d4f4', '#f032e6']
    
    fg_selected = folium.FeatureGroup(name='Selected Stations (Green)')
    fg_unselected = folium.FeatureGroup(name='Unselected Stations (Red)')
    fg_demand = folium.FeatureGroup(name='Demand Points (Blue)')
    fg_unique = folium.FeatureGroup(name='Unique Flows (Orange)', show=False)

    flow_counts = [[0] * max_nodes for _ in range(max_nodes)]
    for F in F_list:
        for i in range(max_nodes):
            for j in range(max_nodes):
                if F[i][j] > 0:
                    flow_counts[i][j] += 1

    for idx, F in enumerate(F_list):
        s_idx = idx + 1
        fg_flows = folium.FeatureGroup(name=f'Shuffle {s_idx} Flows', show=(s_idx == 1))
        current_color = colors[idx % len(colors)]

        for i in range(max_nodes):
            row = df_d1.iloc[i]
            lat, lon = row['lat'], row['lon']
            
            if idx == 0:
                if best_x[i] == 1:
                    used = sum(F[k][i] for k in range(max_nodes))
                    served_list = [f"D{k}({F[k][i]})" for k in range(max_nodes) if F[k][i] > 0]
                    tt = f"Station {i}: {row['name']}<br>Used: {used}/{B}<br>Serves: {', '.join(served_list) or 'None'}"
                    folium.CircleMarker(location=[lat, lon], radius=7, color='green', fill=True, tooltip=tt).add_to(fg_selected)
                else:
                    folium.CircleMarker(location=[lat, lon], radius=3, color='red', fill=True, tooltip=f"Station {i}: {row['name']} (Off)").add_to(fg_unselected)
                
                if D[i] > 0:
                    met = sum(F[i][k] for k in range(max_nodes))
                    st_list = [f"S{k}({F[i][k]})" for k in range(max_nodes) if F[i][k] > 0]
                    tt = f"Demand {i}: {row['name']}<br>Need: {D[i]}<br>Met: {met}<br>Unmet: {D[i]-met}<br>Furthest distance: {Z[i]}<br>Stations: {', '.join(st_list) or 'None'}"
                    folium.CircleMarker(location=[lat, lon], radius=6, color='blue', fill=True, tooltip=tt).add_to(fg_demand)

            for j in range(max_nodes):
                if F[i][j] > 0:
                    folium.PolyLine(
                        locations=[[df_d1.loc[j, 'lat'], df_d1.loc[j, 'lon']], [df_d1.loc[i, 'lat'], df_d1.loc[i, 'lon']]],
                        color=current_color, weight=2, opacity=0.6, 
                        tooltip=f"Shuffle {s_idx}<br>Source: Station {j}<br>Sink: Demand {i}<br>Flow: {F[i][j]}<br> Distance: {L[i][j]}"
                    ).add_to(fg_flows)

                    if flow_counts[i][j] == 1:
                        folium.PolyLine(
                            locations=[[df_d1.loc[j, 'lat'], df_d1.loc[j, 'lon']], [df_d1.loc[i, 'lat'], df_d1.loc[i, 'lon']]],
                            color='orange', weight=4, opacity=0.9, dash_array='5, 5',
                            tooltip=f"Unique Flow (Shuffle {s_idx})<br>Source: Station {j}<br>Sink: Demand {i}<br>Flow: {F[i][j]}<br> Distance: {L[i][j]}"
                        ).add_to(fg_unique)

        fg_flows.add_to(m)

    fg_selected.add_to(m)
    fg_unselected.add_to(m)
    fg_demand.add_to(m)
    fg_unique.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    cache_folder.mkdir(parents=True, exist_ok=True)
    m.save(cache_folder / 'map_output.html')

if __name__ == "__main__":
    data = read_input(input_path)
    if data:
        N, B, C, P, L, R, Z, D = data
        file = choose_solution_file()
        if file:
            with open(file, 'r', encoding='utf-8') as f:
                sol = json.load(f)
            x = sol['best_solution']['x']
            config = sol['metadata']['configuration']
            rng = random.Random(config.get('random_seed', 42))
            model = CustomerRouting(N, B, C, P, R, L, Z, D, config)
            F_list = []
            for _ in range(config.get('num_shuffles', 0)):
                order = list(range(N))
                rng.shuffle(order)
                F_list.append(model.route(x, order))
            generate_flow_map(x, N, B, D, F_list)