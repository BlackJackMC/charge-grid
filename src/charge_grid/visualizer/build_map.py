import json
import pandas as pd
import folium
import random
import requests
import numpy as np
from pathlib import Path

from charge_grid.utils import read_input, E, O, INPUT_DIR, OUTPUT_DIR, METADATA_DIR
from charge_grid.models.cluster import ClusterRouting
from charge_grid.models.customer import CustomerRouting
from charge_grid.models.station import StationRouting



MODEL_MAP = {
    'ClusterRouting': ClusterRouting,
    'CustomerRouting': CustomerRouting,
    'StationRouting': StationRouting,
}

cache_folder = Path('../cache')
current_dir = Path(__file__).parent
input_path = INPUT_DIR / 'input_hcm.txt'
csv_metadata_path = METADATA_DIR / 'data_hcm.csv'
asset_path = current_dir / 'assets'


CSS_PATH = asset_path / 'map_style.css'
JS_PATH = asset_path / 'map_script.js'

def choose_solution_file():
    if not OUTPUT_DIR.exists(): 
        return None
    json_files = list(OUTPUT_DIR.glob('solution_*.json'))
    if not json_files: 
        return None
    
    print("\n--- DANH SÁCH KẾT QUẢ ĐÃ LƯU ---")
    for idx, file_path in enumerate(json_files):
        print(f"[{idx}] {file_path.name}")
        
    try:
        choice = int(input("\nNhập số thứ tự file muốn vẽ bản đồ: "))
        return json_files[choice] if 0 <= choice < len(json_files) else None
    except ValueError: 
        return None

def build_map_data(best_x, F_matrix, df_meta, N, D, L, C, P, R, config):
    map_data = {
        'stations': {}, 
        'unselected_stations': [], 
        'customers': {}, 
        'routes': [],
        'heat_data': []
    }
    
    for i in range(N):
        if D[i] > 0 and i < len(df_meta):
            demand_met = sum(F_matrix[i])
            visited_stations = []
            for j in range(N):
                if F_matrix[i][j] > 0 and j < len(df_meta):
                    visited_stations.append({
                        'id': j, 'name': df_meta.iloc[j]['name'],
                        'batt': int(F_matrix[i][j]), 'distance': float(L[i][j])
                    })

            map_data['customers'][i] = {
                'id': i, 'name': df_meta.iloc[i]['name'],
                'lat': df_meta.iloc[i]['lat'], 'lon': df_meta.iloc[i]['lon'],
                'total_demand': int(D[i]), 'demand_met': int(demand_met),
                'unmet_demand': int(D[i] - demand_met),
                'visited_stations': visited_stations
            }
            map_data['heat_data'].append([df_meta.iloc[i]['lat'], df_meta.iloc[i]['lon'], float(D[i])])

    for j in range(N):
        if j < len(df_meta):
            if best_x[j] == 1:
                served_customers = []
                total_batt = 0
                station_dissat = 0
                
                for i in range(N):
                    if F_matrix[i][j] > 0 and i < len(df_meta):
                        served_customers.append({
                            'id': i, 'name': df_meta.iloc[i]['name'], 
                            'batt': int(F_matrix[i][j]), 'distance': float(L[i][j])
                        })
                        total_batt += F_matrix[i][j]
                        station_dissat += config['beta'] * F_matrix[i][j] * L[i][j]
                
                if total_batt > 0:
                    station_profit = (total_batt * P) - (C + R[j])
                    map_data['stations'][j] = {
                        'id': j, 'name': df_meta.iloc[j]['name'],
                        'lat': df_meta.iloc[j]['lat'], 'lon': df_meta.iloc[j]['lon'],
                        'total_batt': int(total_batt), 'dissatisfaction': float(station_dissat),
                        'profit': float(station_profit), 'customers': served_customers
                    }
            else:
                map_data['unselected_stations'].append({
                    'id': j, 'name': df_meta.iloc[j]['name'],
                    'lat': df_meta.iloc[j]['lat'], 'lon': df_meta.iloc[j]['lon']
                })

    return map_data

def append_osrm_routes(map_data, F_matrix, df_meta, N, L, session):
    print("  > Đang fetch API giao thông thực tế từ OSRM (vui lòng đợi)...")
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
                    'coords': route_coords, 'batt': int(F_matrix[i][j]), 'distance': float(L[i][j])
                })
    return map_data

def generate_interactive_map(best_x, df_meta, N, B, C, P, R, Z, D, L, config, model, shuffle_seed=42):
    cache_folder.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    num_stations = sum(best_x)
    print(f"\n--- Đang xử lý bản đồ tương tác ---")
    
    if hasattr(model, 'solve_flow'):
        F_matrix, _ = model.solve_flow(best_x)
    else:
        rng = random.Random(shuffle_seed)
        current_order = list(range(N))
        rng.shuffle(current_order)
        F_matrix = model.route(best_x, current_order)
        
    F_matrix = np.array(F_matrix) 
    
    total_profit_val, _, _ = E(best_x, F_matrix, C, P, R)
    total_dissat_val, _, _ = O(F_matrix, D, L, config['alpha'], config['beta'])
    
    map_data = build_map_data(best_x, F_matrix, df_meta, N, D, L, C, P, R, config)
    map_data = append_osrm_routes(map_data, F_matrix, df_meta, N, L, session)

    center_lat, center_lon = df_meta['lat'].mean(), df_meta['lon'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='CartoDB positron', prefer_canvas=True)

    try:
        with open(CSS_PATH, 'r', encoding='utf-8') as f:
            css_code = f.read()
    except FileNotFoundError:
        css_code = "/* CSS File missing */"
        print(f"Cảnh báo: Không tìm thấy {CSS_PATH}")
        
    try:
        with open(JS_PATH, 'r', encoding='utf-8') as f:
            js_code = f.read()
    except FileNotFoundError:
        js_code = "// JS File missing"
        print(f"Cảnh báo: Không tìm thấy {JS_PATH}")

    custom_html = f"""
    <script src="https://leaflet.github.io/Leaflet.heat/dist/leaflet-heat.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css"/>
    <style>{css_code}</style>
    
    <div class="glass-panel main-stats-box">
        <div style="color: #17a2b8;"><i class="fas fa-chart-line"></i> Lợi nhuận (E): {total_profit_val:,.2f}</div>
        <div style="color: #dc3545;"><i class="fas fa-frown"></i> Dissatisfaction (O): {total_dissat_val:,.2f}</div>
        <div style="color: #28a745;"><i class="fas fa-charging-station"></i> Trạm đã đặt: {num_stations} / {N} node</div>
    </div>
    <div class="glass-panel" id="info-panel"></div>

    <script>
        // Inject dynamic data before executing custom script
        window.map_data = {json.dumps(map_data)};
        
        // Execute the dynamically loaded external javascript
        {js_code}
    </script>
    """
    
    m.get_root().html.add_child(folium.Element(custom_html))

    map_filename = cache_folder / 'map_interactive_output.html'
    m.save(str(map_filename))
    print(f"  > Hoàn tất! Đã lưu map tại: {map_filename}")
if __name__ == "__main__":
    problem_data = read_input(input_path)
    if problem_data:
        N, B, C, P, L, R, Z, D = problem_data
        
        file_path = choose_solution_file()
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as f:
                sol = json.load(f)
                
            best_x = sol['best_solution']['x']
            saved_config = sol['metadata']['configuration']
            
            # ---> THÊM 2 DÒNG NÀY VÀO ĐÂY <---
            if 'num_shuffles' not in saved_config:
                saved_config['num_shuffles'] = 1
            # ----------------------------------
            
            model_builder_name = saved_config.get('model_builder', 'CustomerRouting')
            ModelClass = MODEL_MAP.get(model_builder_name, CustomerRouting)
            
            model = ModelClass(N, B, C, P, R, L, Z, D, saved_config)
            
            try:
                df_meta = pd.read_csv(csv_metadata_path, nrows=N)
            except Exception as e:
                print(f"Lỗi đọc file CSV: {e}")
                df_meta = None
                
            if df_meta is not None:
                generate_interactive_map(best_x, df_meta, N, B, C, P, R, Z, D, L, saved_config, model)