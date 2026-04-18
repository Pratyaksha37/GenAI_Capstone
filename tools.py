import numpy as np
import joblib
import os
import math
from rag import get_market_insights

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.joblib"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.joblib"))


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return 6371 * 2 * math.asin(math.sqrt(a))

def predict_price(input_data):
    tenure_leasehold = 1 if input_data["tenure"] == "LEASEHOLD" else 0
    prop_detached = 1 if input_data["propertyType"] == "DETACHED" else 0
    prop_semi_detached = 1 if input_data["propertyType"] == "SEMI_DETACHED" else 0
    prop_terraced = 1 if input_data["propertyType"] == "TERRACED" else 0

    # Dynamic Feature Derivations
    energy_map = {'A':7, 'B':6, 'C':5, 'D':4, 'E':3, 'F':2, 'G':1}
    energy_rating_score = energy_map.get(input_data.get('currentEnergyRating', 'D'), 4)
    
    total_rooms = input_data['bedrooms'] + input_data['bathrooms'] + input_data['livingRooms']
    total_rooms = total_rooms if total_rooms > 0 else 1
    sqm_per_room = input_data['floorAreaSqM'] / total_rooms
    
    distance_to_center = haversine(input_data['latitude'], input_data['longitude'], 51.5074, -0.1278)

    features = np.array([[
        input_data["latitude"], input_data["longitude"],
        input_data["bedrooms"], input_data["bathrooms"],
        input_data["floorAreaSqM"], input_data["livingRooms"],
        energy_rating_score, sqm_per_room, distance_to_center,
        tenure_leasehold, prop_detached, prop_semi_detached, prop_terraced
    ]])

    features_scaled = scaler.transform(features)
    predicted_log_price = model.predict(features_scaled)
    return np.exp(predicted_log_price[0])


def get_market_data(query):
    return get_market_insights(query)
