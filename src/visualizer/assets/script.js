document.addEventListener("DOMContentLoaded", function() {
    function initCustomMap() {
        if (typeof L === 'undefined') { setTimeout(initCustomMap, 100); return; }
        var leaflet_map = null;
        for (var key in window) { if (key.startsWith("map_") && window[key] && window[key].addLayer) { leaflet_map = window[key]; break; } }
        if (!leaflet_map) { setTimeout(initCustomMap, 100); return; }

        var data = window.map_data;
        
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

        setTimeout(() => {
            var miniMap = L.map('minimap-div', {
                zoomControl: false, attributionControl: false,
                dragging: false, touchZoom: false, scrollWheelZoom: false, doubleClickZoom: false
            });
            L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png').addTo(miniMap);
            
            if(typeof L.heatLayer !== 'undefined') {
                var heat = L.heatLayer(data.heat_data, {
                    radius: 18, blur: 15, maxZoom: 13, max: 150,
                    gradient: {0.3: 'blue', 0.5: 'cyan', 0.7: 'lime', 0.8: 'yellow', 1.0: 'red'}
                }).addTo(miniMap);
            }

            leaflet_map.on('move', function() {
                miniMap.setView(leaflet_map.getCenter(), Math.max(1, leaflet_map.getZoom() - 3));
            });
            miniMap.setView(leaflet_map.getCenter(), Math.max(1, leaflet_map.getZoom() - 3));
        }, 300);


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

        data.routes.forEach(r => {
            var calcWeight = Math.min(8, Math.max(2, r.batt * 0.05)); 
            
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

            var polylineFlow = L.polyline(r.coords, {
                color: '#38bdf8', 
                weight: Math.max(2, calcWeight - 1), 
                opacity: 0.8, 
                pane: 'routesPane', 
                className: 'flow-line' 
            }).addTo(routeGroup);
            
            polylineFlow.cust_id = r.cust_id; 
            polylineFlow.stat_id = r.stat_id; 
            polylineFlow.isFlow = true;
            routeLayers.push(polylineFlow);
        });

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

        if(data.unselected_stations) {
            data.unselected_stations.forEach(us => {
                var marker = L.circleMarker([us.lat, us.lon], {
                    radius: 5, color: '#dc3545', weight: 1.5, fillColor: '#ffc107', fillOpacity: 0.8
                }).bindTooltip(`Trạm không được chọn: ${us.name}`).addTo(unselectedGroup);
                unselectedMarkers[us.id] = marker;
            });
        }

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