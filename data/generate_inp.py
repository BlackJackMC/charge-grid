import osmnx as ox
import networkx as nx
import pandas as pd
import math
import geopandas as gpd # IMPORTANT: Added GeoPandas for spatial operations
import random

# ==============================
# 1. Đọc CSV
# ==============================
print("Loading locations...")
gdf = pd.read_csv("data_q1.csv")

# ==============================
# 2. Load graph
# ==============================
place = "District 1, Ho Chi Minh City, Vietnam"
print("Downloading graph...")
G = ox.graph_from_place(place, network_type="walk")

# ==============================
# 3. Project CRS (QUAN TRỌNG)
# ==============================
G = ox.project_graph(G)

# ==============================
# 4. Lấy nodes GeoDataFrame
# ==============================
nodes, edges = ox.graph_to_gdfs(G)

# ==============================
# 5. Map lat/lon → CRS của graph (CORRECTED)
# ==============================
print("Projecting CSV coordinates to Graph CRS...")

# Bước A: Chuyển Pandas DataFrame thành GeoDataFrame (chuẩn GPS EPSG:4326)
gdf_geo = gpd.GeoDataFrame(
    gdf, 
    geometry=gpd.points_from_xy(gdf['lon'], gdf['lat']),
    crs="EPSG:4326"
)

# Bước B: Project sang CRS của graph
gdf_proj = gdf_geo.to_crs(nodes.crs)

# ==============================
# 6. Dùng ox.nearest_nodes (NHANH + CHUẨN)
# ==============================
print("Mapping to nearest nodes...")

gdf_proj["road_node"] = ox.distance.nearest_nodes(
    G,
    X=gdf_proj.geometry.x,
    Y=gdf_proj.geometry.y
)

# ==============================
# 7. Tính shortest path (distance chuẩn)
# ==============================
print("\nCalculating shortest paths (optimized)...")


# danh sách road nodes
road_nodes = gdf_proj["road_node"].tolist()
ids = gdf_proj["id"].tolist() # Giả sử file CSV của bạn có cột 'id'

N = len(gdf)

L = [[0]*N for _ in range(N)]

for i in range(N):
    source = road_nodes[i]

    # chạy Dijkstra 1 lần
    lengths = nx.single_source_dijkstra_path_length(
        G,
        source,
        weight="length"
    )

    for j in range(N):
        if i == j:
            L[i][j] = 0
        else:
            target = road_nodes[j]

            if target in lengths:
                dist = lengths[target]

                # fallback nếu dist = 0
                if dist == 0:
                    dist = math.sqrt(
                        (gdf_proj.geometry.x[i] - gdf_proj.geometry.x[j])**2 +
                        (gdf_proj.geometry.y[i] - gdf_proj.geometry.y[j])**2
                    )

                L[i][j] = round(dist, 4)
            

# ==============================
# 8. EXPORT CSV RIÊNG
# ==============================
# print("\nExporting CSVs...")

# # nodes mapped
# nodes_out = gdf.copy()
# nodes_out["road_node"] = gdf_proj["road_node"]
# nodes_out.to_csv("mapped_nodes.csv", index=False)

# # edges
# edges_df = pd.DataFrame(edges_list, columns=["u", "v", "distance"])
# edges_df.to_csv("edges.csv", index=False)

# print("CSV exported!")

# ==============================
# 9. Tạo input.txt
# ==============================

C = 1000
B = 10
P = 5

# demand 1 giá trị / node 
W = [random.randint(5, 30) for _ in range(N)]
R = [round(random.uniform(100.0, 500.0), 2) for _ in range(N)]

with open("input_q1.txt", "w") as f:
    f.write(f"{N} {B} {C} {P}\n")

    # ma trận khoảng cách
    for i in range(N):
        f.write(" ".join(map(str, L[i])) + "\n")

    f.write(" ".join(map(str, R)) + "\n")
    f.write(" ".join(map(str, W)))

print("Done!")