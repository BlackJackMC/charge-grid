import osmnx as ox
import networkx as nx
import pandas as pd
import math
import geopandas as gpd
import random

from charge_grid.utils import INPUT_DIR


print("Loading locations...")
gdf = pd.read_csv("data_hcm.csv")

place = "Ho Chi Minh City, Vietnam"
print("Downloading graph...")
G = ox.graph_from_place(place, network_type="walk")

G = ox.project_graph(G)

nodes, edges = ox.graph_to_gdfs(G)

print("Projecting CSV coordinates to Graph CRS...")

gdf_geo = gpd.GeoDataFrame(
    gdf, 
    geometry=gpd.points_from_xy(gdf['lon'], gdf['lat']),
    crs="EPSG:4326"
)

gdf_proj = gdf_geo.to_crs(nodes.crs)

print("Mapping to nearest nodes...")

gdf_proj["road_node"] = ox.distance.nearest_nodes(
    G,
    X=gdf_proj.geometry.x,
    Y=gdf_proj.geometry.y
)

print("\nCalculating shortest paths (optimized)...")


road_nodes = gdf_proj["road_node"].tolist()
ids = gdf_proj["id"].tolist()

N = len(gdf)

L = [[0]*N for _ in range(N)]

for i in range(N):
    source = road_nodes[i]

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

                if dist == 0:
                    dist = math.sqrt(
                        (gdf_proj.geometry.x[i] - gdf_proj.geometry.x[j])**2 +
                        (gdf_proj.geometry.y[i] - gdf_proj.geometry.y[j])**2
                    )

                L[i][j] = round(dist, 4)
            

# print("\nExporting CSVs...")

# # nodes mapped
# nodes_out = gdf.copy()
# nodes_out["road_node"] = gdf_proj["road_node"]
# nodes_out.to_csv("mapped_nodes.csv", index=False)

# # edges
# edges_df = pd.DataFrame(edges_list, columns=["u", "v", "distance"])
# edges_df.to_csv("edges.csv", index=False)

# print("CSV exported!")


C = 325
B = 220
P = 9

# Demand D
# D = []

# for _ in range(N):
#     if random.random() < 0.6:   # 60% have demand
#         D.append(random.randint(20, 80))
#     else:
        # D.append(0)
# D = [random.randint(50, 100) for _ in range(N)]

districts = gdf_proj["district"].tolist()
district_D_range = {
    "District 1": (60, 100),
    "District 3": (55, 95),
    "District 5": (50, 90),
    "District 10": (50, 90),

    "Binh Thanh District": (40, 80),
    "Phu Nhuan District": (40, 80),
    "Tan Binh District": (35, 70),
    "Go Vap District": (30, 60),
    "Tan Phu District": (25, 55),
    "Binh Tan District": (25, 50),

    "District 4": (30, 70),
    "District 6": (25, 60),
    "District 7": (20, 50),
    "District 8": (20, 50),
    "District 12": (20, 40)
}
D = []

for d in districts:
    if random.random() < 0.6:
        low, high = district_D_range.get(d, (20, 60))
        D.append(random.randint(low, high))
    else:
        D.append(0)

R = [random.choice([30, 40]) for _ in range(N)]

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

with open(INPUT_DIR / "input_hcm.txt", "w") as f:
    f.write(f"{N} {B} {C} {P}\n")

    for i in range(N):
        f.write(" ".join(map(str, L[i])) + "\n")

    f.write(" ".join(map(str, R)) + "\n")
    f.write(" ".join(map(str, Z)) + "\n")
    f.write(" ".join(map(str, D)))

print("Done!")