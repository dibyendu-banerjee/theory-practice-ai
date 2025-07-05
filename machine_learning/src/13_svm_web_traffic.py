"""# **Chapter 7:Ensemble Methods and Random Forest**

# Use Case: Eco-friendly Transportation Routes: Optimizing logistics and delivery routes to reduce fuel consumption and greenhouse gas emissions.
"""

!pip install emoji


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from prettytable import PrettyTable
import emoji

# Simulating a sample dataset for transportation routes
data = {
    'traffic': np.random.randint(1, 10, 1000),  # Traffic intensity (1-10 scale)
    'weather': np.random.randint(1, 5, 1000),   # Weather (1: Clear, 2: Light Rain, etc.)
    'road_type': np.random.randint(1, 4, 1000), # Road types (1: Highway, 2: Urban, 3: Rural)
    'distance': np.random.randint(5, 50, 1000), # Distance in km
    'vehicle_type': np.random.randint(1, 4, 1000), # Vehicle (1: Electric, 2: Hybrid, 3: Gas)
    'fuel_consumption': np.random.uniform(5, 15, 1000), # Liters per 100km
    'route_time': np.random.uniform(20, 120, 1000) # Time in minutes
}

df = pd.DataFrame(data)

# Features and target
X = df[['traffic', 'weather', 'road_type', 'distance', 'vehicle_type', 'fuel_consumption']]
y = df['route_time']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Test cases
test_cases = [
    {"traffic": 7, "weather": 2, "road_type": 1, "distance": 30, "vehicle_type": 2, "fuel_consumption": 10},
    {"traffic": 4, "weather": 1, "road_type": 3, "distance": 50, "vehicle_type": 3, "fuel_consumption": 12},
    {"traffic": 8, "weather": 3, "road_type": 2, "distance": 40, "vehicle_type": 1, "fuel_consumption": 8},
]

# Mappings
vehicle_types = {1: "Electric", 2: "Hybrid", 3: "Gas"}
weather_conditions = {1: "Clear", 2: "Light Rain", 3: "Heavy Rain", 4: "Snow"}

# Emoji maps (using verified emoji names)
weather_icons = {
    1: emoji.emojize(":sun:", language='alias'),
    2: emoji.emojize(":cloud_rain:", language='alias'),
    3: emoji.emojize(":cloud_with_rain:", language='alias'),
    4: emoji.emojize(":cloud_snow:", language='alias')
}

traffic_icons = {
    1: emoji.emojize(":red_circle:", language='alias'),
    2: emoji.emojize(":orange_circle:", language='alias'),
    3: emoji.emojize(":yellow_circle:", language='alias'),
    4: emoji.emojize(":green_circle:", language='alias'),
    5: emoji.emojize(":blue_circle:", language='alias'),
    6: emoji.emojize(":purple_circle:", language='alias'),
    7: emoji.emojize(":brown_circle:", language='alias'),
    8: emoji.emojize(":black_circle:", language='alias'),
    9: emoji.emojize(":white_circle:", language='alias')
}

# Table
table = PrettyTable()
table.field_names = [
    "Test Case", "Traffic", "Weather", "Road Type", "Distance (km)",
    "Vehicle Type", "Fuel Consumption", "Weather Icon", "Traffic Icon", "Predicted Time (mins)"
]

# Loop through test cases
for idx, case in enumerate(test_cases, start=1):
    traffic = case["traffic"]
    weather = case["weather"]
    road_type = case["road_type"]
    distance = case["distance"]
    vehicle_type = case["vehicle_type"]
    fuel_consumption = case["fuel_consumption"]

    input_data = np.array([[traffic, weather, road_type, distance, vehicle_type, fuel_consumption]])
    input_scaled = scaler.transform(input_data)
    predicted_time = model.predict(input_scaled)[0]

    table.add_row([
        f"Case {idx}",
        traffic,
        weather_conditions.get(weather, "Unknown"),
        road_type,
        distance,
        vehicle_types.get(vehicle_type, "Unknown"),
        f"{fuel_consumption:.2f}",
        weather_icons.get(weather, ""),
        traffic_icons.get(traffic, ""),
        f"{predicted_time:.2f}"
    ])

print("\n--- Eco-friendly Route Prediction for 3 Test Cases ---")
print(table)

# Feature importance plot
feature_importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, feature_importances, color='lightblue')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance for Route Time Prediction')
plt.grid(True)
plt.tight_layout()
plt.show()