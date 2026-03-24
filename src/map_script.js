document.addEventListener("DOMContentLoaded", function() {
    setTimeout(function() {
        var leaflet_map = null;
        for (var key in window) {
            if (key.startsWith("map_") && window[key] instanceof L.Map) {
                leaflet_map = window[key]; break;
            }
        }

        if (leaflet_map && window.map_data) {
            var data = window.map_data;
            
            // 1. CHỈNH Z-INDEX PANE (Dòng đường đi nằm dưới trạm/khách hàng)
            leaflet_map.createPane('routesPane');
            leaflet_map.getPane('routesPane').style.zIndex = 390;

            // 2. LAYER GROUPS CHO CÁI TOGGLE PANEL
            var routeGroup = L.layerGroup().addTo(leaflet_map);
            var stationGroup = L.layerGroup().addTo(leaflet_map);
            var customerGroup = L.layerGroup().addTo(leaflet_map);
            var unselectedGroup = L.layerGroup().addTo(leaflet_map);
            // BIẾN LƯU TRẠNG THÁI HIỆN TẠI ĐỂ RESET ANIMATION
            var activeCustomerMarker = null;
            var activeStationMarker = null;

            // 3. VẼ ĐƯỜNG ĐI
            data.routes.forEach(r => {
                var calcWeight = Math.min(12, Math.max(2, r.batt * 0.02)); 
                var polyline = L.polyline(r.coords, {
                    color: '#3186cc', weight: calcWeight, opacity: 0.4, 
                    dashArray: '5, 10', pane: 'routesPane'
                }).addTo(routeGroup);
                
                polyline.cust_id = r.cust_id; 
                polyline.stat_id = r.stat_id; 
                polyline.baseWeight = calcWeight;
                routeLayers.push(polyline);
            });

            // 4. VẼ TRẠM SẠC (Icon thu nhỏ 20%)
            var stationIcon = L.divIcon({
                className: 'custom-station-icon',
                html: '<div class="station-icon-inner"><i class="fas fa-battery-full"></i></div>',
                iconSize: [26, 26], 
                iconAnchor: [13, 13]
            });

            Object.values(data.stations).forEach(s => {
                var marker = L.marker([s.lat, s.lon], {icon: stationIcon, zIndexOffset: 1000}).addTo(stationGroup);
                marker.on('click', function(e) {
                    L.DomEvent.stopPropagation(e);
                    
                    if (activeStationMarker && activeStationMarker._icon) { L.DomUtil.removeClass(activeStationMarker._icon.firstChild, 'station-glow'); }
                    activeStationMarker = marker;
                    if (marker._icon) { L.DomUtil.addClass(marker._icon.firstChild, 'station-glow'); }
                    
                    if (activeCustomerMarker) {
                        activeCustomerMarker.setStyle({radius: 8, weight: 1.5});
                        if(activeCustomerMarker._path) L.DomUtil.removeClass(activeCustomerMarker._path, 'customer-glow');
                        activeCustomerMarker = null;
                    }

                    highlightRoutes(null, s.id);
                    showStationInfo(s);
                });
            });

            // 5. VẼ KHÁCH HÀNG (Kích thước gốc r=8)
            Object.values(data.customers).forEach(c => {
                var nodeColor = c.unmet_demand > 0 ? '#ff8c00' : '#007bff';
                var marker = L.circleMarker([c.lat, c.lon], {
                    radius: 8, color: 'white', weight: 1.5, fillColor: nodeColor, fillOpacity: 0.9
                }).addTo(customerGroup);
                
                marker.on('click', function(e) {
                    L.DomEvent.stopPropagation(e);
                    
                    if (activeCustomerMarker) { 
                        activeCustomerMarker.setStyle({radius: 8, weight: 1.5}); 
                        if(activeCustomerMarker._path) L.DomUtil.removeClass(activeCustomerMarker._path, 'customer-glow');
                    }
                    activeCustomerMarker = marker;
                    marker.setStyle({radius: 14, weight: 3}); // Bơm to node lên
                    if(marker._path) L.DomUtil.addClass(marker._path, 'customer-glow'); // Phát sáng

                    if (activeStationMarker && activeStationMarker._icon) { 
                        L.DomUtil.removeClass(activeStationMarker._icon.firstChild, 'station-glow'); 
                        activeStationMarker = null;
                    }
                    
                    if (data.unselected_stations) {
                        data.unselected_stations.forEach(us => {
                            L.circleMarker([us.lat, us.lon], {
                                radius: 4, color: '#999', weight: 1, fillColor: '#ccc', fillOpacity: 0.6
                            }).bindTooltip(`Trạm không chọn: ${us.name}`).addTo(unselectedGroup);
                        });
                    }

                    highlightRoutes(c.id, null);
                    showCustomerInfo(c);
                });
            });

            // 6. CONTROL PANEL (Góc trái dưới)
            var overlayMaps = {
                "<i class='fas fa-battery-full' style='color:#28a745'></i> Trạm đổi pin": stationGroup,
                "<i class='fas fa-users' style='color:#007bff'></i> Khách hàng": customerGroup,
                "<i class='fas fa-times-circle' style='color:#999'></i> Trạm không chọn": unselectedGroup, // THÊM DÒNG NÀY
                "<i class='fas fa-route' style='color:#3186cc'></i> Tuyến đường": routeGroup
            };
            L.control.layers(null, overlayMaps, {position: 'bottomleft', collapsed: false}).addTo(leaflet_map);

            // 7. CHÚ THÍCH (Góc phải dưới)
            var legend = L.control({position: 'bottomright'});
            legend.onAdd = function (map) {
                var div = L.DomUtil.create('div', 'legend');
                div.innerHTML = `
                    <h4><i class="fas fa-info-circle"></i> Chú thích</h4>
                    <div><span class="legend-icon" style="background:#28a745;"><i class="fas fa-battery-full"></i></span> Trạm đổi pin đặt tại node</div>
                    <div><span class="legend-dot" style="background:#007bff;"></span> Khách hàng (Đã đủ pin)</div>
                    <div><span class="legend-dot" style="background:#ff8c00;"></span> Khách hàng (Còn thiếu pin)</div>
                    <div><span class="legend-line"></span> Tuyến phân phối pin</div>
                `;
                return div;
            };
            legend.addTo(leaflet_map);

            // 8. LOGIC HIGHLIGHT ĐƯỜNG ĐI (To lên, đổi màu sáng, kiến bò)
            function highlightRoutes(custId, statId) {
                routeLayers.forEach(layer => {
                    if ((custId !== null && layer.cust_id === custId) || (statId !== null && layer.stat_id === statId)) {
                        layer.setStyle({opacity: 1.0, color: '#ff7f50', weight: layer.baseWeight + 5});
                        if (layer._path) L.DomUtil.addClass(layer._path, 'route-active');
                        layer.bringToFront();
                    } else {
                        layer.setStyle({opacity: 0.1, color: '#999', weight: layer.baseWeight});
                        if (layer._path) L.DomUtil.removeClass(layer._path, 'route-active');
                    }
                });
            }

            // 9. LOGIC RESET KHI CLICK CHỖ TRỐNG
            window.resetMapUI = function() {
                routeLayers.forEach(layer => { 
                    layer.setStyle({opacity: 0.4, color: '#3186cc', weight: layer.baseWeight}); 
                    if (layer._path) L.DomUtil.removeClass(layer._path, 'route-active');
                });
                
                if (activeCustomerMarker) {
                    activeCustomerMarker.setStyle({radius: 8, weight: 1.5});
                    if(activeCustomerMarker._path) L.DomUtil.removeClass(activeCustomerMarker._path, 'customer-glow');
                    activeCustomerMarker = null;
                }
                
                if (activeStationMarker && activeStationMarker._icon) {
                    L.DomUtil.removeClass(activeStationMarker._icon.firstChild, 'station-glow');
                    activeStationMarker = null;
                }
                
                document.getElementById('info-panel').classList.remove('panel-open');
            };
            leaflet_map.on('click', window.resetMapUI);

            // BẢNG THÔNG TIN TRẠM
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
                s.customers.forEach(c => { html += `<li><b>${c.name}</b> <span style="color:#007bff; font-weight:bold;">${c.batt} pin</span></li>`; });
                html += `</ul>`;
                panel.innerHTML = html;
                panel.classList.add('panel-open');
            }

            // BẢNG THÔNG TIN KHÁCH HÀNG
            function showCustomerInfo(c) {
                var panel = document.getElementById('info-panel');
                var statusBadge = c.unmet_demand > 0 ? `<span style="background:#ff8c00;color:white;padding:3px 8px;border-radius:4px;font-size:12px;margin-left:10px;box-shadow:0 2px 4px rgba(0,0,0,0.2);">Còn thiếu pin</span>` : `<span style="background:#007bff;color:white;padding:3px 8px;border-radius:4px;font-size:12px;margin-left:10px;box-shadow:0 2px 4px rgba(0,0,0,0.2);">Đủ pin</span>`;
                var titleColor = c.unmet_demand > 0 ? '#ff8c00' : '#007bff';
                var bgWarning = c.unmet_demand > 0 ? '#fff3cd' : '#f8f9fa';

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
                    <div class="stat-row" style="background-color: ${bgWarning}; border: 1px solid ${c.unmet_demand > 0 ? '#ffeeba' : 'transparent'};">
                        <b><i class="fas fa-exclamation-triangle" style="color:#ff8c00;"></i> Unmet Demand (Thiếu):</b> <span style="color:#ff8c00; font-weight:bold;">${c.unmet_demand}</span>
                    </div>
                    ${visitedHtml}
                `;
                panel.innerHTML = html;
                panel.classList.add('panel-open');
            }
        }
    }, 500);
});