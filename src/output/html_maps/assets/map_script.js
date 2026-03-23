document.addEventListener("DOMContentLoaded", function() {
    setTimeout(function() {
        var leaflet_map = null;
        for (var key in window) {
            if (key.startsWith("map_") && window[key] instanceof L.Map) {
                leaflet_map = window[key]; break;
            }
        }

        if (leaflet_map && window.map_data) {
            var routeLayers = [];
            var data = window.map_data;
            
            data.routes.forEach(r => {
                var calcWeight = Math.min(12, Math.max(2, r.batt * 0.02)); 
                var polyline = L.polyline(r.coords, {
                    color: '#3186cc', weight: calcWeight, opacity: 0.3, dashArray: '5, 10'
                }).addTo(leaflet_map);
                polyline.cust_id = r.cust_id; polyline.stat_id = r.stat_id; polyline.baseWeight = calcWeight;
                routeLayers.push(polyline);
            });

            var stationIcon = L.divIcon({
                className: 'custom-station-icon',
                html: '<div style="background-color: #28a745; color: white; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 2px solid white; box-shadow: 0 3px 6px rgba(0,0,0,0.4); font-size: 15px;"><i class="fas fa-battery-full"></i></div>',
                iconSize: [32, 32], 
                iconAnchor: [16, 16]
            });

            Object.values(data.stations).forEach(s => {
                var marker = L.marker([s.lat, s.lon], {icon: stationIcon, zIndexOffset: 1000}).addTo(leaflet_map);
                marker.on('click', function(e) {
                    L.DomEvent.stopPropagation(e);
                    highlightRoutes(null, s.id);
                    showStationInfo(s);
                });
            });

            Object.values(data.customers).forEach(c => {
                var nodeColor = c.unmet_demand > 0 ? '#ff8c00' : '#007bff';
                var marker = L.circleMarker([c.lat, c.lon], {
                    radius: 10, color: 'white', weight: 1.5, fillColor: nodeColor, fillOpacity: 0.9
                }).addTo(leaflet_map);
                marker.on('click', function(e) {
                    L.DomEvent.stopPropagation(e);
                    highlightRoutes(c.id, null);
                    showCustomerInfo(c);
                });
            });

            function highlightRoutes(custId, statId) {
                routeLayers.forEach(layer => {
                    if ((custId !== null && layer.cust_id === custId) || (statId !== null && layer.stat_id === statId)) {
                        layer.setStyle({opacity: 1.0, color: '#ff7f50', weight: layer.baseWeight + 3, dashArray: null});
                        layer.bringToFront();
                    } else {
                        layer.setStyle({opacity: 0.05, color: '#999', weight: layer.baseWeight, dashArray: '5, 10'});
                    }
                });
            }

            window.resetMapUI = function() {
                routeLayers.forEach(layer => { layer.setStyle({opacity: 0.3, color: '#3186cc', weight: layer.baseWeight, dashArray: '5, 10'}); });
                document.getElementById('info-panel').style.display = 'none';
            };
            leaflet_map.on('click', window.resetMapUI);

            function showStationInfo(s) {
                var panel = document.getElementById('info-panel');
                var html = `
                    <div class="panel-header">
                        <h3 style="color:#28a745;"><i class="fas fa-bolt"></i> ${s.name}</h3>
                        <button class="close-btn" onclick="window.resetMapUI()">&times;</button>
                    </div>
                    <div class="stat-row"><b><i class="fas fa-battery-full text-success"></i> Số pin cung cấp:</b> ${s.total_batt}</div>
                    <div class="stat-row"><b><i class="fas fa-frown text-danger"></i> Dissatisfaction (O):</b> ${s.dissatisfaction.toFixed(2)}</div>
                    <h4 style="margin: 15px 0 10px 0;"><i class="fas fa-users"></i> Khách hàng phục vụ:</h4>
                    <ul class="customer-list">
                `;
                s.customers.forEach(c => { html += `<li><b>${c.name}</b> <span>${c.batt} pin</span></li>`; });
                html += `</ul>`;
                panel.innerHTML = html;
                panel.style.display = 'block';
            }

            function showCustomerInfo(c) {
                var panel = document.getElementById('info-panel');
                var statusBadge = c.unmet_demand > 0 ? `<span style="background:#ff8c00;color:white;padding:3px 8px;border-radius:4px;font-size:12px;margin-left:10px;">Còn thiếu pin</span>` : `<span style="background:#007bff;color:white;padding:3px 8px;border-radius:4px;font-size:12px;margin-left:10px;">Đủ pin</span>`;
                var titleColor = c.unmet_demand > 0 ? '#ff8c00' : '#007bff';
                var bgWarning = c.unmet_demand > 0 ? '#fff3cd' : '#f8f9fa';

                // TẠO DANH SÁCH CÁC TRẠM ĐÃ GHÉ
                var visitedHtml = "";
                if (c.visited_stations && c.visited_stations.length > 0) {
                    visitedHtml = `<h4 style="margin: 15px 0 10px 0;"><i class="fas fa-charging-station"></i> Các trạm đã đổi pin:</h4>
                                   <ul class="customer-list">`;
                    c.visited_stations.forEach(st => {
                        visitedHtml += `<li><b>${st.name}</b> <span style="color:#28a745; font-weight:bold;">${st.batt} pin</span></li>`;
                    });
                    visitedHtml += `</ul>`;
                } else {
                    visitedHtml = `<div style="margin-top:15px; font-style:italic; color:#888;"><i class="fas fa-sad-tear"></i> Khách hàng chưa đổi được pin ở trạm nào.</div>`;
                }

                var html = `
                    <div class="panel-header">
                        <h3 style="color:${titleColor};"><i class="fas fa-user"></i> Khách hàng</h3>
                        <button class="close-btn" onclick="window.resetMapUI()">&times;</button>
                    </div>
                    <div style="font-size: 16px; margin-bottom: 15px; display:flex; align-items:center;">
                        <b>${c.name}</b> ${statusBadge}
                    </div>
                    <div class="stat-row"><b><i class="fas fa-battery-empty text-secondary"></i> Nhu cầu ban đầu:</b> ${c.total_demand}</div>
                    <div class="stat-row"><b><i class="fas fa-sync-alt text-primary"></i> Đã đổi được:</b> ${c.demand_met}</div>
                    <div class="stat-row" style="background-color: ${bgWarning};">
                        <b><i class="fas fa-exclamation-triangle" style="color:#ff8c00;"></i> Unmet Demand (Thiếu):</b> <span style="color:#ff8c00; font-weight:bold;">${c.unmet_demand}</span>
                    </div>
                    ${visitedHtml}
                `;
                panel.innerHTML = html;
                panel.style.display = 'block';
            }
        }
    }, 500);
});