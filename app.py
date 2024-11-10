import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS for handling cross-origin requests

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the dataset and preprocess
file_path = './dataset.xlsx'
data = pd.read_excel(file_path)

# Data cleaning
damage_column = 'Total Damage, Adjusted (\'000 US$)'  # Check the exact column name
if damage_column in data.columns:
    data[damage_column].fillna(data[damage_column].median(), inplace=True)
else:
    print(f"Column '{damage_column}' not found. Please verify the column name.")

# Mock environmental indicators (Rainfall, Soil Moisture, Terrain Change)
data['Rainfall'] = np.random.uniform(0, 500, data.shape[0])
data['Soil Moisture'] = np.random.uniform(0, 1, data.shape[0])
data['Terrain Change'] = np.random.uniform(0, 100, data.shape[0])

# Feature scaling
scaler = StandardScaler()
data[['Rainfall', 'Soil Moisture', 'Terrain Change']] = scaler.fit_transform(
    data[['Rainfall', 'Soil Moisture', 'Terrain Change']]
)

# Create labels for Flood and Landslide Risk for demonstration purposes
data['Flood Risk'] = ((data['Rainfall'] > 0.7) & (data['Soil Moisture'] > 0.5)).astype(int)
data['Landslide Risk'] = ((data['Terrain Change'] > 0.7) & (data['Soil Moisture'] > 0.6)).astype(int)

# Train the models on the initial dataset
flood_model = RandomForestClassifier(n_estimators=100, random_state=42)
landslide_model = RandomForestClassifier(n_estimators=100, random_state=42)

features = data[['Rainfall', 'Soil Moisture', 'Terrain Change']]
flood_labels = data['Flood Risk']
landslide_labels = data['Landslide Risk']

flood_model.fit(features, flood_labels)
landslide_model.fit(features, landslide_labels)

# Define the API endpoint
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print("Data received:", data)  # Log received data
    
    try:
        # Ensure that data fields are valid
        rainfall = float(data['rainfall'])
        soil_moisture = float(data['soilMoisture'])
        terrain_changes = float(data['terrainChanges'])
    except (KeyError, ValueError) as e:
        print(f"Error: {e}")
        return jsonify({"error": "Invalid input data"}), 400

    # Standardize real-time data
    real_time_df = pd.DataFrame([[rainfall, soil_moisture, terrain_changes]],
                                 columns=['Rainfall', 'Soil Moisture', 'Terrain Change'])
    standardized_data = scaler.transform(real_time_df)

    # Predict flood and landslide risks
    flood_risk_pred = flood_model.predict(standardized_data)[0]
    landslide_risk_pred = landslide_model.predict(standardized_data)[0]

    # Map predictions to 'High' or 'Low' labels
    flood_risk = "High" if flood_risk_pred == 1 else "Low"
    landslide_risk = "High" if landslide_risk_pred == 1 else "Low"

    # Return the results as a JSON response
    return jsonify({
        'floodRisk': flood_risk,
        'landslideRisk': landslide_risk
    })

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
