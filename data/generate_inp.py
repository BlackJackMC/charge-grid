import osmnx as ox
import pandas as pd
import networkx as nx
import math
import random

# ==============================
# 1. Đọc danh sách location
# ==============================

print("Loading locations from data.csv...")

gdf = pd.read_csv("data.csv")

print("Total locations:", len(gdf))

# ==============================
# 2. Lấy road network
# ==============================

place = "District 1, Ho Chi Minh City, Vietnam"

print("\nDownloading road network...")

G = ox.graph_from_place(place, network_type="drive")

# ==============================
# 3. Lấy road nodes
# ==============================

nodes = list(G.nodes(data=True))

# ==============================
# 4. Hàm tìm node gần nhất
# ==============================

def nearest_node(lat, lon):

    best_node = None
    best_dist = float("inf")

    for node_id, data in nodes:

        node_lat = data["y"]
        node_lon = data["x"]

        dist = math.sqrt(
            (lat - node_lat) ** 2 +
            (lon - node_lon) ** 2
        )

        if dist < best_dist:
            best_dist = dist
            best_node = node_id

    return best_node

# ==============================
# 5. Map location → road node
# ==============================

print("\nMapping locations to road nodes...")

road_nodes = []

for _, row in gdf.iterrows():

    rn = nearest_node(row["lat"], row["lon"])
    road_nodes.append(rn)

gdf["road_node"] = road_nodes

# ==============================
# 6. Tính shortest path
# ==============================

print("\nCalculating shortest paths...")

edges = []

for i in range(len(gdf)):
    for j in range(i + 1, len(gdf)):

        u = gdf.iloc[i]["road_node"]
        v = gdf.iloc[j]["road_node"]

        try:

            dist = nx.shortest_path_length(
                G,
                u,
                v,
                weight="length"
            )

            edges.append(
                (
                    gdf.iloc[i]["id"],
                    gdf.iloc[j]["id"],
                    round(dist, 2)
                )
            )

        except:
            continue

# ==============================
# 7. Tạo tham số bài toán
# ==============================

D = len(gdf)
Y = len(edges)

S = 50   # dung lượng pin
T = 24   # số time step

C = 1000  # cost xây trạm
B = 10    # số pin sạc mỗi bước
P = 5     # profit mỗi pin

# tạo demand Wi ngẫu nhiên

W = []

for _, row in gdf.iterrows():

    a = row["amenity"]

    if a == "university":
        demand = random.randint(15, 30)

    elif a == "school":
        demand = random.randint(12, 25)

    elif a == "college":
        demand = random.randint(10, 22)

    elif a == "parking":
        demand = random.randint(8, 18)

    elif a == "fuel":
        demand = random.randint(5, 12)

    elif a == "charging_station":
        demand = random.randint(3, 10)

    else:
        demand = random.randint(5, 15)

    W.append(demand)

# ==============================
# 8. Ghi file input.txt
# ==============================

print("\nWriting input.txt...")

with open("input.txt", "w") as f:

    # dòng 1
    f.write(f"{D} {Y} {S} {T}\n")

    # dòng 2
    f.write(f"{C} {B} {P}\n")

    # edges
    for u, v, d in edges:
        f.write(f"{u} {v} {d}\n")

    # demand
    f.write(" ".join(map(str, W)))

print("Done! File input.txt created.")