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


gdf["lat"] = gdf.geometry.centroid.y
gdf["lon"] = gdf.geometry.centroid.x

# lấy khoảng 50 location
# gdf = gdf.head(50)

# đánh số node
gdf["id"] = range(1, len(gdf) + 1)

data = gdf[["id", "name", "amenity", "lat", "lon"]]

data.to_csv("data.csv", index=False)

print("Saved to data.csv")

# ==============================
# 3. In ra màn hình
# ==============================

for _, row in data.iterrows():
    print(f"{row['id']}. {row['name']} | {row['amenity']} | lat={row['lat']:.6f}, lon={row['lon']:.6f}")