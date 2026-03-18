import osmnx as ox
import pandas as pd
import networkx as nx
import math

# ==============================
# 1. Take data from OSM
# ==============================

place = "District 1, Ho Chi Minh City, Vietnam"

tags = {
    "amenity": [
        "parking",
        "fuel",
        "charging_station",
        "school",
        "university",
        "college",
        "hospital",
        "mall"
    ],
    "shop": [
        "supermarket",
        "convenience"
    ],
    "highway": [
        "bus_stop"
    ],
    "leisure": [
        "park"
    ]

}

print("Downloading locations...")

gdf = ox.features_from_place(place, tags)
gdf = gdf.reset_index()

gdf = gdf[["name", "amenity", "geometry"]]

gdf["name"] = gdf["name"].fillna("Unknown")


gdf["lat"] = gdf.geometry.centroid.y
gdf["lon"] = gdf.geometry.centroid.x

#About 50 location
#gdf = gdf.head(100)

# Numberable data
gdf["id"] = range(1, len(gdf) + 1)

data = gdf[["id", "name", "amenity", "lat", "lon"]]


data.to_csv("data_q1.csv", index=False)


print("Saved to data_q1.csv")



# ==============================
# 2. Print to console
# ==============================

#for _, row in data.iterrows():
    #print(f"{row['id']}. {row['name']} | {row['amenity']} | lat={row['lat']:.6f}, lon={row['lon']:.6f}")