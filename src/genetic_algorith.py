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

def route(x, station_order=None) -> list[list[int]]:
    station_battery = {j: B for j in range(N) if x[j] == 1}

    F = [[0 for _ in range(N)] for _ in range(N)]

    if station_order is None:
        station_order = list(range(N))
        random.shuffle(station_order)

    local_D = list(D)

    for j in station_order:
        if x[j] == 1:
            for i in precomputed_nearest[j]:
                if local_D[i] > 0 and station_battery[j] > 0:
                    served = min(local_D[i], station_battery[j])
                    F[i][j] = served
                    local_D[i] -= served
                    station_battery[j] -= served
                if station_battery[j] == 0:
                    break

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
        F = config['behavior_model'](x, station_order=order)
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
        F = config['behavior_model'](current_x, station_order=order) # FIXED: pass station_order instead of demand_order
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
    for j in range(N):
        sorted_demands = sorted(range(N), key=lambda i: L[i][j])
        valid_demands = [i for i in sorted_demands if L[i][j] <= Z[i]]
        precomputed_nearest.append(valid_demands)
    
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
    import requests
    import json
    
    map_folder = output_folder / 'html_maps'
    map_folder.mkdir(parents=True, exist_ok=True)
    map_rng = random.Random(42)
    five_orders = []
    
    for _ in range(5):
        order = list(range(N))
        map_rng.shuffle(order)
        five_orders.append(order)
        
    # Dùng session để gọi API OSRM nhanh hơn
    session = requests.Session()

    for idx, current_order in enumerate(five_orders):
        print(f"\n--- Đang xử lý Bản đồ {idx + 1}/5 ---")
        F_matrix = config['behavior_model'](best_x, station_order=current_order)
        
        # 1. Tính Tổng Lợi Nhuận (Profit - E) cho bản đồ hiện tại
        total_profit = E(best_x, F_matrix)
        
        center_lat = df_meta['lat'].mean()
        center_lon = df_meta['lon'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='CartoDB positron')

        # 2. Tạo Data Structure để đẩy sang Javascript Frontend
        map_data = {
            'stations': {},
            'customers': {},
            'routes': []
        }
        
        # Gom thông tin Khách Hàng
        for i in range(N):
            if sum(F_matrix[i]) > 0 and i < len(df_meta):
                map_data['customers'][i] = {
                    'id': i,
                    'name': df_meta.iloc[i]['name'],
                    'lat': df_meta.iloc[i]['lat'],
                    'lon': df_meta.iloc[i]['lon'],
                    'demand_met': sum(F_matrix[i])
                }

        # Gom thông tin Trạm Sạc & Tính Dissatisfaction trạm
        for j in range(N):
            if best_x[j] == 1 and j < len(df_meta):
                served_customers = []
                total_batt = 0
                station_dissat = 0
                
                for i in range(N):
                    if F_matrix[i][j] > 0 and i < len(df_meta):
                        served_customers.append({
                            'id': i,
                            'name': df_meta.iloc[i]['name'],
                            'batt': F_matrix[i][j]
                        })
                        total_batt += F_matrix[i][j]
                        # Dissatisfaction phần khoảng cách
                        station_dissat += config['beta'] * F_matrix[i][j] * L[i][j]
                
                if total_batt > 0: # Chỉ hiển thị trạm có phục vụ
                    map_data['stations'][j] = {
                        'id': j,
                        'name': df_meta.iloc[j]['name'],
                        'lat': df_meta.iloc[j]['lat'],
                        'lon': df_meta.iloc[j]['lon'],
                        'total_batt': total_batt,
                        'dissatisfaction': station_dissat,
                        'customers': served_customers
                    }

        # Gom Routes và gọi API OSRM để lấy đường đi thực tế
        print("  > Đang lấy đường đi giao thông thực tế từ OSRM (sẽ mất một chút thời gian)...")
        for i in range(N):
            for j in range(N):
                if F_matrix[i][j] > 0 and i < len(df_meta) and j < len(df_meta):
                    lat_i, lon_i = df_meta.iloc[i]['lat'], df_meta.iloc[i]['lon']
                    lat_j, lon_j = df_meta.iloc[j]['lat'], df_meta.iloc[j]['lon']
                    
                    # Mặc định là đường thẳng chim bay (fallback)
                    route_coords = [[lat_i, lon_i], [lat_j, lon_j]]
                    
                    try:
                        # Gọi API đường thực tế (OSRM dùng format lon,lat)
                        url = f"http://router.project-osrm.org/route/v1/driving/{lon_i},{lat_i};{lon_j},{lat_j}?overview=full&geometries=geojson"
                        res = session.get(url, timeout=2).json()
                        if res.get('code') == 'Ok':
                            # Leaflet cần [lat, lon] nên ta phải đảo ngược tọa độ trả về
                            route_coords = [[c[1], c[0]] for c in res['routes'][0]['geometry']['coordinates']]
                    except Exception:
                        pass # Nếu API lỗi/quá tải, dùng lại đường thẳng chim bay

                    map_data['routes'].append({
                        'cust_id': i,
                        'stat_id': j,
                        'coords': route_coords,
                        'batt': F_matrix[i][j]
                    })

        # 3. Code HTML/CSS/JS tuỳ chỉnh quản lý tương tác bản đồ
        custom_html = f"""
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css"/>
        <style>
            #info-panel {{
                position: fixed;
                top: 20px;
                right: 20px;
                width: 330px;
                background: white;
                padding: 20px;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
                z-index: 9999;
                max-height: 80vh;
                overflow-y: auto;
                font-family: 'Segoe UI', Arial, sans-serif;
                display: none;
            }}
            .total-profit-box {{
                position: fixed;
                top: 20px;
                left: 60px;
                background: #17a2b8;
                color: white;
                padding: 12px 20px;
                border-radius: 8px;
                z-index: 9999;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-weight: bold;
                font-size: 16px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.2);
                border: 2px solid #117a8b;
            }}
            .panel-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 2px solid #eee;
                padding-bottom: 10px;
                margin-bottom: 15px;
            }}
            .panel-header h3 {{ margin: 0; font-size: 18px; line-height: 1.4; }}
            .close-btn {{ background: none; border: none; font-size: 24px; cursor: pointer; color: #888; }}
            .close-btn:hover {{ color: #dc3545; }}
            .stat-row {{ margin-bottom: 8px; font-size: 14px; padding: 6px; background: #f8f9fa; border-radius: 5px; }}
            .customer-list {{ list-style-type: none; padding: 0; margin: 0; font-size: 13px; }}
            .customer-list li {{ background: #f1f3f5; margin-bottom: 6px; padding: 10px; border-radius: 6px; border-left: 4px solid #007bff; display:flex; justify-content: space-between; }}
        </style>

        <div class="total-profit-box"><i class="fas fa-chart-line"></i> Lợi nhuận tổng (E): {total_profit:,.2f}</div>
        <div id="info-panel"></div>

        <script>
            var map_data = {json.dumps(map_data)};
            var leaflet_map = null;

            document.addEventListener("DOMContentLoaded", function() {{
                setTimeout(function() {{
                    // Tìm instance thực sự của bản đồ Leaflet do Folium tạo ra
                    for (var key in window) {{
                        if (key.startsWith("map_") && window[key] instanceof L.Map) {{
                            leaflet_map = window[key];
                            break;
                        }}
                    }}

                    if (leaflet_map) {{
                        var routeLayers = [];
                        
                        // A. Vẽ tất cả các Routes
                        map_data.routes.forEach(r => {{
                            var calWeight = Math.min(12, Math.max(2, r.batt * 0.2))
                            var polyline = L.polyline(r.coords, {{
                                color: '#3186cc',
                                weight: calWeight,
                                opacity: 0.3,
                                dashArray: '5, 10'
                            }}).addTo(leaflet_map);
                            polyline.cust_id = r.cust_id;
                            polyline.stat_id = r.stat_id;
                            polyline.baseWeight = calWeight;
                            routeLayers.push(polyline);
                        }});

                        // B. Định dạng Icon Trạm Sạc
                        var stationIcon = L.divIcon({{
                            className: 'custom-station-icon',
                            html: '<div style="background-color: #28a745; color: white; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 2px solid white; box-shadow: 0 3px 6px rgba(0,0,0,0.4); font-size: 15px;"><i class="fas fa-charging-station"></i></div>',
                            iconSize: [32, 32],
                            iconAnchor: [16, 16]
                        }});

                        // C. Vẽ Node Trạm Sạc
                        Object.values(map_data.stations).forEach(s => {{
                            var marker = L.marker([s.lat, s.lon], {{icon: stationIcon, zIndexOffset: 1000}}).addTo(leaflet_map);
                            marker.on('click', function(e) {{
                                L.DomEvent.stopPropagation(e); // Ngăn không cho map bắt sự kiện
                                highlightRoutes(null, s.id);
                                showStationInfo(s);
                            }});
                        }});

                        // D. Vẽ Node Khách Hàng
                        Object.values(map_data.customers).forEach(c => {{
                            var marker = L.circleMarker([c.lat, c.lon], {{
                                radius: 5, color: 'white', weight: 1, fillColor: '#dc3545', fillOpacity: 0.9
                            }}).addTo(leaflet_map);
                            marker.on('click', function(e) {{
                                L.DomEvent.stopPropagation(e);
                                highlightRoutes(c.id, null);
                                showCustomerInfo(c);
                            }});
                        }});

                        // Logic Highlight đường đi
                        function highlightRoutes(custId, statId) {{
                            routeLayers.forEach(layer => {{
                                if ((custId !== null && layer.cust_id === custId) ||
                                    (statId !== null && layer.stat_id === statId)) {{
                                    layer.setStyle({{opacity: 1.0, color: '#ff7f50', weight: layer.baseWeight + 3, dashArray: null}});
                                    layer.bringToFront();
                                }} else {{
                                    layer.setStyle({{opacity: 0.05, color: '#999', weight: layer.baseWeight, dashArray: '5, 10'}});
                                }}
                            }});
                        }}

                        // Xoá highlight khi click ra chỗ trống
                        window.resetMapUI = function() {{
                            routeLayers.forEach(layer => {{
                                layer.setStyle({{opacity: 0.3, color: '#3186cc', weight: layer.baseWeight, dashArray: '5, 10'}});
                            }});
                            document.getElementById('info-panel').style.display = 'none';
                        }};
                        leaflet_map.on('click', window.resetMapUI);

                        // Bảng UI Trạm
                        function showStationInfo(s) {{
                            var panel = document.getElementById('info-panel');
                            var html = `
                                <div class="panel-header">
                                    <h3 style="color:#28a745;"><i class="fas fa-bolt"></i> ${{s.name}}</h3>
                                    <button class="close-btn" onclick="window.resetMapUI()">&times;</button>
                                </div>
                                <div class="stat-row"><b><i class="fas fa-battery-full text-success"></i> Số pin cung cấp:</b> ${{s.total_batt}}</div>
                                <div class="stat-row"><b><i class="fas fa-frown text-danger"></i> Dissatisfaction (O):</b> ${{s.dissatisfaction.toFixed(2)}}</div>
                                <h4 style="margin: 15px 0 10px 0;"><i class="fas fa-users"></i> Khách hàng phục vụ:</h4>
                                <ul class="customer-list">
                            `;
                            s.customers.forEach(c => {{
                                html += `<li><b>${{c.name}}</b> <span>${{c.batt}} pin</span></li>`;
                            }});
                            html += `</ul>`;
                            panel.innerHTML = html;
                            panel.style.display = 'block';
                        }}

                        // Bảng UI Khách Hàng
                        function showCustomerInfo(c) {{
                            var panel = document.getElementById('info-panel');
                            var html = `
                                <div class="panel-header">
                                    <h3 style="color:#dc3545;"><i class="fas fa-user"></i> Khách hàng</h3>
                                    <button class="close-btn" onclick="window.resetMapUI()">&times;</button>
                                </div>
                                <div style="font-size: 16px; margin-bottom: 10px;"><b>${{c.name}}</b></div>
                                <div class="stat-row"><b><i class="fas fa-sync-alt text-primary"></i> Tổng pin đã đổi:</b> ${{c.demand_met}}</div>
                            `;
                            panel.innerHTML = html;
                            panel.style.display = 'block';
                        }}
                    }}
                }}, 500); // Đợi Leaflet map load xong mới thực thi
            }});
        </script>
        """
        
        m.get_root().html.add_child(folium.Element(custom_html))

        map_filename = map_folder / f"Map_Result_Seed42_Shuffle_{idx + 1}.html"
        m.save(str(map_filename))
        print(f"  > Đã lưu map thành công: {map_filename}")