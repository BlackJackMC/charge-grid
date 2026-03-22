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

C = 325
B = 55
P = 9

# Demand D
D = []

for _ in range(N):
    if random.random() < 0.6:   # 60% có nhu cầu
        D.append(random.randint(20, 80))
    else:
        D.append(0)
# D = [random.randint(50, 100) for _ in range(N)]

# districts = gdf_proj["district"].tolist()
# district_D_range = {
#     "District 1": (80, 100),
#     "District 3": (75, 95),
#     "District 5": (70, 90),
#     "District 10": (70, 90),

#     "Binh Thanh District": (60, 85),
#     "Phu Nhuan District": (50, 70),
#     "Tan Binh District": (55, 65),
#     "Go Vap District": (45, 70),
#     "Tan Phu District": (45, 70),
#     "Binh Tan District": (40, 65),

#     "District 6": (40, 70),
#     "District 8": (35, 65),
#     "District 12": (30, 50),
# }

# def gen_D(district):
#     low, high = district_D_range.get(district, (20, 60))  # default
#     return random.randint(low, high)

# D = [gen_D(d) for d in districts]

# Rental cost R
R = [random.choice([30, 40]) for _ in range(N)]

# Max travel distance Z
Z = []

for i in range(N):
    valid_dist = [L[i][j] for j in range(N) if j != i and L[i][j] < 1e8]

    if valid_dist:
        min_dist = min(valid_dist)

        if min_dist < 100:
            factor = random.uniform(5.0, 10.0)
        elif min_dist < 500:
            factor = random.uniform(2.0, 4.0)
        elif min_dist < 1000:
            factor = random.uniform(1.8, 2.5)
        else:
            factor = random.uniform(1.2, 1.5)

    Z.append(round(min_dist * factor, 2))

# ==============================
# 10. EXPORT INPUT.TXT
# ==============================
with open("input_q1.txt", "w") as f:
    f.write(f"{N} {B} {C} {P}\n")

    for i in range(N):
        f.write(" ".join(map(str, L[i])) + "\n")

    f.write(" ".join(map(str, R)) + "\n")
    f.write(" ".join(map(str, Z)) + "\n")
    f.write(" ".join(map(str, D)))

print("Done!")