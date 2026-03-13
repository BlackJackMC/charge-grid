import osmnx as ox
import pandas as pd
import networkx as nx
import math

# ==============================
# 1. Lấy danh sách địa điểm
# ==============================

place = "District 1, Ho Chi Minh City, Vietnam"

tags = {
    "amenity": [
        "parking",
        "fuel",
        "charging_station",
        "school",
        "university",
        "college"
    ]
}

print("Downloading locations...")

gdf = ox.features_from_place(place, tags)
gdf = gdf.reset_index()

gdf = gdf[["name", "amenity", "geometry"]]

gdf["name"] = gdf["name"].fillna("Unknown")

# lấy tọa độ
gdf["lat"] = gdf.geometry.centroid.y
gdf["lon"] = gdf.geometry.centroid.x

# lấy khoảng 50 location
# gdf = gdf.head(50)

# đánh số node
gdf["id"] = range(1, len(gdf) + 1)

print("\nLocations:\n")

for _, row in gdf.iterrows():
    print(f"{row['id']}. {row['name']} | {row['amenity']} | lat={row['lat']:.6f}, lon={row['lon']:.6f}")
# ==============================
# 2. Lấy road network
# ==============================

print("\nDownloading road network...")

G = ox.graph_from_place(place, network_type="drive")

# ==============================
# 3. Lấy danh sách road nodes
# ==============================

nodes = list(G.nodes(data=True))

# hàm tìm node gần nhất
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


# map location → road node
road_nodes = []

for _, row in gdf.iterrows():
    rn = nearest_node(row["lat"], row["lon"])
    road_nodes.append(rn)

gdf["road_node"] = road_nodes

# ==============================
# 4. Find shortest path
# ==============================

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
# 5. Output
# ==============================

print("\nEdges:\n")

for u, v, d in edges:
    print(u, v, d)

print("\nTotal nodes:", len(gdf))
print("Total edges:", len(edges))