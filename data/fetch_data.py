import osmnx as ox
import pandas as pd
import networkx as nx
import math

from charge_grid.utils import METADATA_DIR

districts = [
    "District 1, Ho Chi Minh City, Vietnam",
    "District 3, Ho Chi Minh City, Vietnam",
    "District 4, Ho Chi Minh City, Vietnam",
    "District 5, Ho Chi Minh City, Vietnam",
    "District 6, Ho Chi Minh City, Vietnam",
    "District 7, Ho Chi Minh City, Vietnam",
    "District 8, Ho Chi Minh City, Vietnam",
    "District 10, Ho Chi Minh City, Vietnam",
    "District 12, Ho Chi Minh City, Vietnam",
    "Binh Thanh District, Ho Chi Minh City, Vietnam",
    "Phu Nhuan District, Ho Chi Minh City, Vietnam",
    "Tan Binh District, Ho Chi Minh City, Vietnam",
    "Go Vap District, Ho Chi Minh City, Vietnam",
    "Tan Phu District, Ho Chi Minh City, Vietnam",
    "Binh Tan District, Ho Chi Minh City, Vietnam"
]

tags = {
    "amenity": [
        "parking",
        "fuel",
        "charging_station",
        "university",
        "college",
        "mall"
    ],
    "shop": [
        "convenience"
    ],
    "leisure": [
        "park"
    ]
}

print("Downloading locations...")

gdfs = []

for place in districts:
    print(f"Fetching {place}...")
    gdf_temp = ox.features_from_place(place, tags)
    gdf_temp["district"] = place.split(",")[0]
    gdfs.append(gdf_temp)

gdf = pd.concat(gdfs, ignore_index=True)
gdf = gdf.reset_index(drop=True)

gdf = gdf[["name", "amenity", "geometry", "district"]]

gdf["name"] = gdf["name"].fillna("Unknown")

gdf["lat"] = gdf.geometry.centroid.y
gdf["lon"] = gdf.geometry.centroid.x

gdf = gdf.drop_duplicates(subset=["lat", "lon"])

gdf["id"] = range(1, len(gdf) + 1)

data = gdf[["id", "name", "amenity", "lat", "lon", "district"]]

data.to_csv(METADATA_DIR / "data_hcm.csv", index=False)
print("Saved to data_hcm.csv")