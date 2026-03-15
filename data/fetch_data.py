import osmnx as ox

place = "Ho Chi Minh City, Vietnam"

tags = {
    "amenity": [
        "parking",
        "fuel",
        "charging_station",
        "school",
        "university",
        "college",
        "hospital",
        "mall",
        "cinema",
        "restaurant"
    ],
    "shop": [
        "supermarket",
        "convenience",
        "department_store",
        "clothes",
        "electronics",
        "mobile_phone",
    ],
    "tourism": [
        "hotel"
    ],
    "highway": [
        "bus_stop"
    ],
    "building": [
        "apartments",
    ],
    "leisure": [
        "park"
    ]

}

print("Downloading locations...")

gdf = ox.features_from_place(place, tags).fillna("Unknown")

gdf = gdf.fillna("Unknown")


gdf["lat"] = gdf.geometry.centroid.y
gdf["lon"] = gdf.geometry.centroid.x

gdf["id"] = range(1, len(gdf) + 1)

data = gdf[["id", "name", "amenity", "lat", "lon"]]

data.to_csv("data.csv", index=False)


print("Saved to data.csv")