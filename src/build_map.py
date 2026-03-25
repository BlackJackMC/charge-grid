import json
import pandas as pd
import folium
import random
import requests
import numpy as np
from pathlib import Path

# IMPORT THEO KIẾN TRÚC MỚI
from utils import read_input, E, O
from models.cluster import ClusterRouting
from models.customer import CustomerRouting
from models.station import StationRouting
from models.behavioral import BehavioralRouting
from models.milp_routing import MILPRoutingORTools
# TỪ ĐIỂN MAP TÊN CLASS
MODEL_MAP = {
    'ClusterRouting': ClusterRouting,
    'CustomerRouting': CustomerRouting,
    'StationRouting': StationRouting,
    'BehavioralRouting': BehavioralRouting,
    'MILPRoutingORTools': MILPRoutingORTools,
}

input_folder = Path('..')
output_folder = Path('./output')
cache_folder = Path('./cache')
input_path = input_folder / 'input_q1.txt'
csv_metadata_path = input_folder / 'data_hcm.csv'

def choose_solution_file():
    if not output_folder.exists(): 
        return None
    json_files = list(output_folder.glob('solution_*.json'))
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
        'heat_data': [] # NEW: Data cho Minimap Heatmap
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
            # Ghi nhận dữ liệu độ nóng (Lat, Lon, Cường độ Demand)
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
        # Dành cho MILP OR-Tools
        F_matrix, _ = model.solve_flow(best_x)
    else:
        # Dành cho các thuật toán Heuristic cũ
        rng = random.Random(shuffle_seed)
        current_order = list(range(N))
        rng.shuffle(current_order)
        F_matrix = model.route(best_x, current_order)
        
    # ĐẢM BẢO CHỈ GIỮ LẠI DÒNG NÀY, KHÔNG GỌI LẠI model.route() NỮA
    F_matrix = np.array(F_matrix) 
    
    # 2. Gọi hàm E, O
    total_profit_val, _, _ = E(best_x, F_matrix, C, P, R)
    total_dissat_val, _, _ = O(F_matrix, D, L, config['alpha'], config['beta'])
    
    map_data = build_map_data(best_x, F_matrix, df_meta, N, D, L, C, P, R, config)
    map_data = append_osrm_routes(map_data, F_matrix, df_meta, N, L, session)

    center_lat, center_lon = df_meta['lat'].mean(), df_meta['lon'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='CartoDB positron', prefer_canvas=True)

    css_code = """
    /* ĐỒNG BỘ DESIGN SYSTEM: GLASSMORPHISM */
    .glass-panel {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
        border: 1px solid rgba(255,255,255,0.6);
        font-family: 'Segoe UI', Arial, sans-serif;
        color: #333;
    }

    /* Info Panel - Góc Phải Trên */
    #info-panel { position: fixed; top: 20px; right: 20px; width: 340px; z-index: 9999; max-height: 80vh; overflow-y: auto; padding: 20px; transform: translateX(120%); opacity: 0; pointer-events: none; transition: transform 0.4s cubic-bezier(0.25, 0.8, 0.25, 1), opacity 0.4s ease; }
    #info-panel.panel-open { transform: translateX(0); opacity: 1; pointer-events: auto; }
    
    /* Stats Box - Góc Trái Trên */
    .main-stats-box { position: fixed; top: 20px; left: 60px; z-index: 9998; padding: 15px 20px; border-left: 5px solid #17a2b8; transition: transform 0.2s, box-shadow 0.2s; }
    .main-stats-box:hover { transform: translateY(-2px); box-shadow: 0 10px 25px rgba(0,0,0,0.15); }
    .main-stats-box div { margin-bottom: 8px; font-size: 15px; font-weight: bold; }
    .main-stats-box div:last-child { margin-bottom: 0; }
    
    /* Panel chung & Thành phần */
    .panel-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #eee; padding-bottom: 10px; margin-bottom: 12px; }
    .panel-header h3 { margin: 0; font-size: 18px; line-height: 1.4; font-weight: bold; }
    .close-btn { background: none; border: none; font-size: 24px; cursor: pointer; color: #888; transition: color 0.2s; }
    .close-btn:hover { color: #dc3545; }
    .panel-badges { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 15px; }
    .badge { padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.15); }
    .badge-blue { background: #007bff; } .badge-orange { background: #ff8c00; } .badge-green { background: #28a745; }
    .stat-row { margin-bottom: 8px; font-size: 14px; padding: 8px 10px; background: #f8f9fa; border-radius: 6px; box-shadow: inset 0 1px 3px rgba(0,0,0,0.03); display: flex; justify-content: space-between; align-items: center;}
    .stat-label { font-weight: 600; color: #555; } .stat-value { font-weight: bold; color: #222; }
    .section-title { margin: 15px 0 10px 0; font-size: 14px; font-weight: bold; color: #444; border-bottom: 1px solid #eee; padding-bottom: 5px; }
    .customer-list { list-style-type: none; padding: 0; margin: 0; font-size: 13px; }
    .customer-list li { background: #f1f3f5; margin-bottom: 6px; padding: 10px; border-radius: 6px; border-left: 4px solid #007bff; display:flex; justify-content: space-between; transition: background 0.2s; align-items: center;}
    .customer-list li:hover { background: #e2e6ea; }
    .list-subtext { font-size: 11px; color: #777; font-weight: normal; display: block; }
    
    /* Legend - Chú thích (Chuyển sang Trái Dưới) */
    .legend-panel { padding: 15px; margin-bottom: 10px !important; margin-left: 10px !important; font-size: 13px;}
    .legend-panel h4 { margin: 0 0 10px 0; font-size: 15px; color: #117a8b; border-bottom: 2px solid #eee; padding-bottom: 5px; font-weight: bold;}
    .legend-icon { display: inline-flex; width: 20px; height: 20px; border-radius: 50%; align-items: center; justify-content: center; border: 1px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3); vertical-align: middle; margin-right: 8px; }
    .legend-icon i { color: white; font-size: 10px; }
    .legend-dot { display: inline-block; width: 12px; height: 12px; border-radius: 50%; border: 1px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3); vertical-align: middle; margin-right: 12px; margin-left: 4px; }
    .legend-line { display: inline-block; width: 20px; height: 4px; background: #ff7f50; vertical-align: middle; margin-right: 8px; border-radius: 2px; border-top: 2px dashed white; }
    .legend-panel div { margin-bottom: 8px; }
    
    /* Ghi đè Leaflet Control Layers - Đồng bộ Glass Panel */
    .leaflet-control-layers { background: rgba(255, 255, 255, 0.95) !important; backdrop-filter: blur(10px) !important; border-radius: 12px !important; box-shadow: 0 6px 20px rgba(0,0,0,0.1) !important; border: 1px solid rgba(255,255,255,0.6) !important; padding: 5px !important; font-family: 'Segoe UI', Arial, sans-serif !important; }
    
    /* Minimap Radar - Góc Phải Dưới */
    .minimap-wrapper { padding: 10px; margin-bottom: 20px !important; margin-right: 20px !important; width: 240px; }
    .minimap-header { font-size: 13px; font-weight: bold; margin-bottom: 8px; color: #444; text-align: center; }
    #minimap-div { height: 180px; width: 100%; border-radius: 8px; border: 1px solid #ddd; background: #111;}

    /* Icons & Animations */
    .station-icon-inner { background-color: #28a745; color: white; width: 26px; height: 26px; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 2px solid white; box-shadow: 0 4px 10px rgba(0,0,0,0.4); font-size: 12px; transition: all 0.3s; }
    .station-icon-inner:hover { transform: scale(1.15) translateY(-3px); box-shadow: 0 8px 15px rgba(0,0,0,0.5); }
    .station-glow { transform: scale(1.3) !important; box-shadow: 0 0 15px 5px rgba(40, 167, 69, 0.8) !important; border: 2px solid #fff; z-index: 9999 !important;}
    .connected-glow-station { transform: scale(1.15) !important; box-shadow: 0 0 12px 3px rgba(23, 162, 184, 0.8) !important; border: 2px solid #fff; }
    path.leaflet-interactive { transition: stroke-width 0.3s ease, stroke 0.3s ease, stroke-opacity 0.3s ease, r 0.3s, fill-opacity 0.3s; }
    .customer-glow { filter: drop-shadow(0px 0px 8px rgba(255, 140, 0, 1)); }
    .connected-glow-cust { filter: drop-shadow(0px 0px 6px rgba(23, 162, 184, 0.9)); stroke-width: 2.5px; stroke: #fff;}
    .flow-line { stroke-dasharray: 8 16; stroke-linecap: round; }
    .route-active-flow { animation: energy-pulse 0.5s linear infinite !important; filter: drop-shadow(0 0 3px rgba(255,255,255,0.8)); }
    @keyframes energy-pulse { 0% { stroke-dashoffset: 24; } 100% { stroke-dashoffset: 0; } }
    .route-tooltip { font-family: 'Segoe UI', Arial; font-size: 13px; padding: 6px 10px; border-radius: 6px; }
    """

    js_code = """
    document.addEventListener("DOMContentLoaded", function() {
        function initCustomMap() {
            if (typeof L === 'undefined') { setTimeout(initCustomMap, 100); return; }
            var leaflet_map = null;
            for (var key in window) { if (key.startsWith("map_") && window[key] && window[key].addLayer) { leaflet_map = window[key]; break; } }
            if (!leaflet_map) { setTimeout(initCustomMap, 100); return; }

            var data = window.map_data;
            
            // 1. TẠO RADAR MINIMAP HEATMAP (Góc Phải Dưới)
            var MinimapControl = L.Control.extend({
                options: { position: 'bottomright' },
                onAdd: function (map) {
                    var div = L.DomUtil.create('div', 'glass-panel minimap-wrapper');
                    div.innerHTML = '<div class="minimap-header"><i class="fas fa-fire text-danger"></i> Phân phối nhu cầu S</div><div id="minimap-div"></div>';
                    L.DomEvent.disableClickPropagation(div);
                    L.DomEvent.disableScrollPropagation(div);
                    return div;
                }
            });
            leaflet_map.addControl(new MinimapControl());

            // Đợi div render xong mới init map con
            setTimeout(() => {
                var miniMap = L.map('minimap-div', {
                    zoomControl: false, attributionControl: false,
                    dragging: false, touchZoom: false, scrollWheelZoom: false, doubleClickZoom: false
                });
                // Nền Dark Mode cho Minimap để tôn lên Heatmap
                L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png').addTo(miniMap);
                
                // Nạp data Heatmap
                if(typeof L.heatLayer !== 'undefined') {
                    var heat = L.heatLayer(data.heat_data, {
                        radius: 18, blur: 15, maxZoom: 13, max: 150,
                        gradient: {0.3: 'blue', 0.5: 'cyan', 0.7: 'lime', 0.8: 'yellow', 1.0: 'red'}
                    }).addTo(miniMap);
                }

                // Đồng bộ Camera
                leaflet_map.on('move', function() {
                    miniMap.setView(leaflet_map.getCenter(), Math.max(1, leaflet_map.getZoom() - 3));
                });
                miniMap.setView(leaflet_map.getCenter(), Math.max(1, leaflet_map.getZoom() - 3));
            }, 300);

            // 2. CHÚ THÍCH LEGEND (Chuyển sang Góc Trái Dưới)
            var legend = L.control({position: 'bottomleft'});
            legend.onAdd = function (map) {
                var div = L.DomUtil.create('div', 'glass-panel legend-panel');
                div.innerHTML = `
                    <h4><i class="fas fa-info-circle"></i> Chú thích</h4>
                    <div><span class="legend-icon" style="background:#28a745;"><i class="fas fa-battery-full"></i></span> Trạm được chọn</div>
                    <div><span class="legend-dot" style="background:#ffc107; border-color:#dc3545;"></span> Trạm không chọn</div>
                    <div><span class="legend-dot" style="background:#007bff;"></span> Khách (Đã đủ pin)</div>
                    <div><span class="legend-dot" style="background:#ff8c00;"></span> Khách (Còn thiếu)</div>
                    <div><span class="legend-line"></span> Dòng chảy (Khách ➔ Trạm)</div>
                `;
                return div;
            };
            legend.addTo(leaflet_map);

            // CÁC LAYER GROUPS CHÍNH
            leaflet_map.createPane('routesPane');
            leaflet_map.getPane('routesPane').style.zIndex = 390;

            var routeGroup = L.layerGroup().addTo(leaflet_map);
            var stationGroup = L.layerGroup().addTo(leaflet_map);
            var customerGroup = L.layerGroup().addTo(leaflet_map);
            var unselectedGroup = L.layerGroup().addTo(leaflet_map);
            
            var routeLayers = [];
            var customerMarkers = {};
            var stationMarkers = {};
            var unselectedMarkers = {};

            var activeCustomerMarker = null;
            var activeStationMarker = null;

            // VẼ TUYẾN ĐƯỜNG (2 LỚP)
            data.routes.forEach(r => {
                var calcWeight = Math.min(8, Math.max(2, r.batt * 0.05)); 
                
                // Lớp 1: Đường nền mờ (The "Pipe")
                var polylineBase = L.polyline(r.coords, {
                    color: '#1e293b', 
                    weight: calcWeight + 2, 
                    opacity: 0.3, 
                    pane: 'routesPane'
                }).addTo(routeGroup);
                
                polylineBase.cust_id = r.cust_id; 
                polylineBase.stat_id = r.stat_id; 
                polylineBase.baseWeight = calcWeight; 
                polylineBase.isBase = true;
                routeLayers.push(polylineBase);

                // Lớp 2: Dòng chảy năng lượng (The "Flow")
                // Chỉnh opacity thành 0.8 để nó hiện luôn trên video
                var polylineFlow = L.polyline(r.coords, {
                    color: '#38bdf8', 
                    weight: Math.max(2, calcWeight - 1), 
                    opacity: 0.8, 
                    pane: 'routesPane', 
                    className: 'flow-line' // Gán class animation
                }).addTo(routeGroup);
                
                polylineFlow.cust_id = r.cust_id; 
                polylineFlow.stat_id = r.stat_id; 
                polylineFlow.isFlow = true;
                routeLayers.push(polylineFlow);
            });

            // VẼ TRẠM SẠC
            var stationIcon = L.divIcon({
                className: 'custom-station-icon',
                html: '<div class="station-icon-inner"><i class="fas fa-battery-full"></i></div>',
                iconSize: [26, 26], iconAnchor: [13, 13]
            });
            Object.values(data.stations).forEach(s => {
                var marker = L.marker([s.lat, s.lon], {icon: stationIcon, zIndexOffset: 1000}).addTo(stationGroup);
                stationMarkers[s.id] = marker;
                marker.bindTooltip(`<b>Trạm ${s.id}: ${s.name}</b><br>Đã cấp: ${s.total_batt} pin`);
                marker.on('click', function(e) {
                    L.DomEvent.stopPropagation(e);
                    window.resetMapUI(); 
                    activeStationMarker = marker;
                    if (marker._icon) L.DomUtil.addClass(marker._icon.firstChild, 'station-glow');
                    highlightNetwork(null, s.id);
                    showStationInfo(s);
                });
            });

            // VẼ TRẠM KHÔNG CHỌN
            if(data.unselected_stations) {
                data.unselected_stations.forEach(us => {
                    var marker = L.circleMarker([us.lat, us.lon], {
                        radius: 5, color: '#dc3545', weight: 1.5, fillColor: '#ffc107', fillOpacity: 0.8
                    }).bindTooltip(`Trạm không được chọn: ${us.name}`).addTo(unselectedGroup);
                    unselectedMarkers[us.id] = marker;
                });
            }

            // VẼ KHÁCH HÀNG
            Object.values(data.customers).forEach(c => {
                var nodeColor = c.unmet_demand > 0 ? '#ff8c00' : '#007bff';
                var marker = L.circleMarker([c.lat, c.lon], {
                    radius: 8, color: 'white', weight: 1.5, fillColor: nodeColor, fillOpacity: 0.9
                }).addTo(customerGroup);
                customerMarkers[c.id] = marker;
                marker.bindTooltip(`<b>Khách ${c.id}: ${c.name}</b><br>Đổi: ${c.demand_met}/${c.total_demand}`);
                marker.on('click', function(e) {
                    L.DomEvent.stopPropagation(e);
                    window.resetMapUI(); 
                    activeCustomerMarker = marker;
                    marker.setStyle({radius: 14, weight: 3, opacity: 1, fillOpacity: 1}); 
                    if(marker._path) L.DomUtil.addClass(marker._path, 'customer-glow');
                    highlightNetwork(c.id, null);
                    showCustomerInfo(c);
                });
            });

            // Đưa vào Layer Control (Góc trái dưới, xếp dưới Legend)
            var overlayMaps = {
                "<i class='fas fa-battery-full' style='color:#28a745'></i> Trạm đổi pin": stationGroup,
                "<i class='fas fa-users' style='color:#007bff'></i> Khách hàng": customerGroup,
                "<i class='fas fa-times-circle' style='color:#dc3545'></i> Trạm không chọn": unselectedGroup,
                "<i class='fas fa-route' style='color:#3186cc'></i> Tuyến đường": routeGroup
            };
            L.control.layers(null, overlayMaps, {position: 'bottomleft', collapsed: false}).addTo(leaflet_map);

            function highlightNetwork(custId, statId) {
                Object.values(customerMarkers).forEach(m => m.setStyle({opacity: 0.2, fillOpacity: 0.2}));
                Object.values(stationMarkers).forEach(m => m.setOpacity(0.3));
                Object.values(unselectedMarkers).forEach(m => m.setStyle({opacity: 0.2, fillOpacity: 0.2}));

                var connectedCusts = new Set();
                var connectedStats = new Set();

                routeLayers.forEach(layer => {
                    var isConnected = false;
                    if (custId !== null && layer.cust_id === custId) { isConnected = true; connectedStats.add(layer.stat_id); } 
                    else if (statId !== null && layer.stat_id === statId) { isConnected = true; connectedCusts.add(layer.cust_id); }

                    if (isConnected) {
                        if (layer.isBase) { layer.setStyle({opacity: 0.9, color: '#ff7f50', weight: layer.baseWeight + 4}); } 
                        else if (layer.isFlow) { layer.setStyle({opacity: 1.0}); if (layer._path) L.DomUtil.addClass(layer._path, 'route-active-flow'); }
                        layer.bringToFront();
                    } else {
                        if (layer.isBase) { layer.setStyle({opacity: 0.05, color: '#999', weight: layer.baseWeight}); } 
                        else if (layer.isFlow) { layer.setStyle({opacity: 0}); if (layer._path) L.DomUtil.removeClass(layer._path, 'route-active-flow'); }
                    }
                });

                if (custId !== null) {
                    if (customerMarkers[custId]) customerMarkers[custId].setStyle({opacity: 1, fillOpacity: 1});
                    connectedStats.forEach(sid => {
                        if (stationMarkers[sid]) {
                            stationMarkers[sid].setOpacity(1);
                            if (stationMarkers[sid]._icon) L.DomUtil.addClass(stationMarkers[sid]._icon.firstChild, 'connected-glow-station');
                        }
                    });
                }
                if (statId !== null) {
                    if (stationMarkers[statId]) stationMarkers[statId].setOpacity(1);
                    connectedCusts.forEach(cid => {
                        if (customerMarkers[cid]) {
                            customerMarkers[cid].setStyle({opacity: 1, fillOpacity: 0.9});
                            if (customerMarkers[cid]._path) L.DomUtil.addClass(customerMarkers[cid]._path, 'connected-glow-cust');
                        }
                    });
                }
            }

            window.resetMapUI = function() {
                routeLayers.forEach(layer => { 
                    if (layer.isBase) { layer.setStyle({opacity: 0.4, color: '#3186cc', weight: layer.baseWeight}); } 
                    else if (layer.isFlow) { layer.setStyle({opacity: 0}); if (layer._path) L.DomUtil.removeClass(layer._path, 'route-active-flow'); }
                });
                Object.values(customerMarkers).forEach(m => {
                    m.setStyle({radius: 8, weight: 1.5, opacity: 1, fillOpacity: 0.9});
                    if(m._path) { L.DomUtil.removeClass(m._path, 'customer-glow'); L.DomUtil.removeClass(m._path, 'connected-glow-cust'); }
                });
                Object.values(stationMarkers).forEach(m => {
                    m.setOpacity(1);
                    if(m._icon) { L.DomUtil.removeClass(m._icon.firstChild, 'station-glow'); L.DomUtil.removeClass(m._icon.firstChild, 'connected-glow-station'); }
                });
                Object.values(unselectedMarkers).forEach(m => m.setStyle({opacity: 1, fillOpacity: 0.8}));
                
                activeCustomerMarker = null; activeStationMarker = null;
                var panel = document.getElementById('info-panel');
                if(panel) panel.classList.remove('panel-open');
            };
            leaflet_map.on('click', window.resetMapUI);

            window.showStationInfo = function(s) {
                var panel = document.getElementById('info-panel');
                var profitClass = s.profit >= 0 ? "text-success" : "text-danger";
                var html = `
                    <div class="panel-header">
                        <h3 style="color:#28a745;"><i class="fas fa-charging-station"></i> ${s.name}</h3>
                        <button class="close-btn" onclick="window.resetMapUI()">&times;</button>
                    </div>
                    <div class="panel-badges"><span class="badge badge-green"><i class="fas fa-check-circle"></i> Đang hoạt động</span></div>
                    <div class="stat-row"><span class="stat-label"><i class="fas fa-battery-full text-success"></i> Đã cấp:</span><span class="stat-value">${s.total_batt}</span></div>
                    <div class="stat-row"><span class="stat-label"><i class="fas fa-dollar-sign text-primary"></i> Tiền lời:</span><span class="stat-value ${profitClass}">${s.profit.toLocaleString('en-US')}</span></div>
                    <div class="stat-row"><span class="stat-label"><i class="fas fa-frown text-danger"></i> Dissatisfaction:</span><span class="stat-value">${s.dissatisfaction.toFixed(2)}</span></div>
                    <div class="section-title"><i class="fas fa-users text-primary"></i> Khách hàng phục vụ (${s.customers.length})</div>
                    <ul class="customer-list" style="border-left-color: #28a745;">
                `;
                s.customers.forEach(c => { html += `<li><div><b>${c.name}</b><span class="list-subtext">L<sub>ij</sub>: ${c.distance.toFixed(2)}</span></div><span style="color:#28a745; font-weight:bold; font-size:14px;">${c.batt} pin</span></li>`; });
                html += `</ul>`;
                panel.innerHTML = html;
                panel.classList.add('panel-open');
            }

            window.showCustomerInfo = function(c) {
                var panel = document.getElementById('info-panel');
                var isDeficit = c.unmet_demand > 0;
                var badgeHtml = isDeficit ? `<span class="badge badge-orange"><i class="fas fa-exclamation-triangle"></i> Thiếu pin</span>` : `<span class="badge badge-blue"><i class="fas fa-check-circle"></i> Đủ pin</span>`;
                var titleColor = isDeficit ? '#ff8c00' : '#007bff';
                
                var visitedHtml = "";
                if (c.visited_stations && c.visited_stations.length > 0) {
                    visitedHtml = `<div class="section-title"><i class="fas fa-charging-station text-success"></i> Đã đổi pin tại (${c.visited_stations.length}) trạm</div><ul class="customer-list">`;
                    c.visited_stations.forEach(st => { visitedHtml += `<li><div><b>${st.name}</b><span class="list-subtext">L<sub>ij</sub>: ${st.distance.toFixed(2)}</span></div><span style="color:#007bff; font-weight:bold; font-size:14px;">${st.batt} pin</span></li>`; });
                    visitedHtml += `</ul>`;
                } else {
                    visitedHtml = `<div style="margin-top:20px; text-align:center; color:#888;"><i class="fas fa-sad-tear fa-2x mb-2"></i><br>Chưa đổi được pin.</div>`;
                }

                var html = `
                    <div class="panel-header">
                        <h3 style="color:${titleColor};"><i class="fas fa-user"></i> ${c.name}</h3>
                        <button class="close-btn" onclick="window.resetMapUI()">&times;</button>
                    </div>
                    <div class="panel-badges">${badgeHtml}</div>
                    <div class="stat-row"><span class="stat-label"><i class="fas fa-battery-empty text-secondary"></i> Nhu cầu gốc:</span><span class="stat-value">${c.total_demand}</span></div>
                    <div class="stat-row"><span class="stat-label"><i class="fas fa-sync-alt text-primary"></i> Đã đổi được:</span><span class="stat-value">${c.demand_met}</span></div>
                    <div class="stat-row" style="background-color: ${isDeficit ? '#fff3cd' : '#f8f9fa'}; border: 1px solid ${isDeficit ? '#ffeeba' : 'transparent'};">
                        <span class="stat-label"><i class="fas fa-exclamation-triangle" style="color:#ff8c00;"></i> Bị thiếu (Unmet):</span><span class="stat-value" style="color:#ff8c00;">${c.unmet_demand}</span>
                    </div>
                    ${visitedHtml}
                `;
                panel.innerHTML = html;
                panel.classList.add('panel-open');
            }
        }
        initCustomMap();
    });
    """

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
        window.map_data = {json.dumps(map_data)};
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