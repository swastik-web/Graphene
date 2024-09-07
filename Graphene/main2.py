import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import geopandas as gpd
import matplotlib.pyplot as plt

# Step 1: Load and Preprocess the Data
satellite_data = pd.read_csv(r'C:\Users\sahoo\Downloads\Graphene\Graphene\mock_satellite_data.csv')
ground_data = pd.read_csv(r'C:\Users\sahoo\Downloads\Graphene\Graphene\mock_ground_truth.csv')

# Step 2: Combine the Datasets
data = pd.merge(satellite_data, ground_data, on=['latitude', 'longitude', 'date'])

# Ensure the merge worked
print("Merged data shape:", data.shape)

# Step 3: Feature Engineering
features = ['satellite_no2', 'temperature', 'humidity', 'population_density', 'proximity_to_roads']

X = data[features]
y = data[['ground_no2']]

# Check for missing values and fill them
print("Missing values in X before:", X.isnull().sum())
X = X.fillna(X.mean())
print("Missing values in y before:", y.isnull().sum())
y = y.fillna(y.mean())

# Ensure no infinite or NaN values are left
print("Any infinite values in X?", np.isfinite(X).all().all())
print("Any infinite values in y?", np.isfinite(y).all())

# Step 4: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Initialize and Train the Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 6: Model Validation
y_pred = rf_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")

# Step 7: Cross-Validation
cv_scores = cross_val_score(rf_model, X, y, cv=3, scoring='neg_mean_squared_error')  # Reduced cv to 3 if small dataset
cv_rmse = np.sqrt(-cv_scores)
print(f"Cross-Validated RMSE: {cv_rmse.mean()}")

# Step 8: Use the Model for Predictions on New Data
new_data = pd.read_csv(r'C:\Users\sahoo\Downloads\Graphene\Graphene\mock_satellite_data.csv')
X_new = new_data[features]

# Ensure new data contains valid longitude and latitude for mapping
print(new_data[['longitude', 'latitude']].head())

predicted_no2 = rf_model.predict(X_new)

# Step 9: Save the Predictions as Fine Spatial Resolution Map
gdf = gpd.GeoDataFrame(new_data, geometry=gpd.points_from_xy(new_data.longitude, new_data.latitude))
gdf['predicted_no2'] = predicted_no2

# Plotting the Predicted NO2 Levels
gdf.plot(column='predicted_no2', cmap='OrRd', legend=True, legend_kwds={'label': "Predicted NO₂ Levels", 'orientation': "horizontal"})
plt.title('Fine Resolution NO₂ Map')
plt.show()

# Step 10: Model Output Validation with Independent Data
independent_data = pd.read_csv(r'C:\Users\sahoo\Downloads\Graphene\Graphene\mock_independent_data.csv')
X_independent = independent_data[features]
y_independent = independent_data['ground_no2']

# Predict on independent data
y_independent_pred = rf_model.predict(X_independent)

# Calculate metrics
independent_rmse = np.sqrt(mean_squared_error(y_independent, y_independent_pred))
independent_r2 = r2_score(y_independent, y_independent_pred)

print(f"Independent Validation RMSE: {independent_rmse}")
print(f"Independent Validation R² Score: {independent_r2}")

# Additional Step: Create subplots for all the graphs
fig, axs = plt.subplots(3, 2, figsize=(15, 18))

# Pie Chart for 'proximity_to_roads' categories
bins = [0, 3, 6, 10]
labels = ['Near', 'Moderate', 'Far']
data['proximity_category'] = pd.cut(data['proximity_to_roads'], bins=bins, labels=labels)
category_counts = data['proximity_category'].value_counts()
axs[0, 0].pie(category_counts, labels=labels, autopct='%1.1f%%', colors=['skyblue', 'orange', 'green'], startangle=140)
axs[0, 0].set_title('Proximity to Roads Distribution')

# Histogram for 'satellite_no2'
axs[0, 1].hist(data['satellite_no2'], bins=20, color='skyblue', edgecolor='black')
axs[0, 1].set_title('Distribution of Satellite NO₂ Levels')
axs[0, 1].set_xlabel('Satellite NO₂')
axs[0, 1].set_ylabel('Frequency')

# Line Chart for 'temperature' over time
axs[1, 0].plot(data['date'], data['temperature'], color='blue')
axs[1, 0].set_title('Temperature Over Time')
axs[1, 0].set_xlabel('Date')
axs[1, 0].set_ylabel('Temperature (K)')
axs[1, 0].tick_params(axis='x', rotation=45)

# Area Chart for 'humidity' over time
axs[1, 1].fill_between(data['date'], data['humidity'], color='lightgreen', alpha=0.5)
axs[1, 1].set_title('Humidity Over Time')
axs[1, 1].set_xlabel('Date')
axs[1, 1].set_ylabel('Humidity (%)')
axs[1, 1].tick_params(axis='x', rotation=45)

# Line Chart with Markers for 'satellite_no2' over time
axs[2, 0].plot(data['date'], data['satellite_no2'], marker='o', color='purple')
axs[2, 0].set_title('Satellite NO₂ Levels Over Time')
axs[2, 0].set_xlabel('Date')
axs[2, 0].set_ylabel('Satellite NO₂')
axs[2, 0].tick_params(axis='x', rotation=45)

# Bar Chart for 'population_density'
axs[2, 1].bar(data.index, data['population_density'], color='orange')
axs[2, 1].set_title('Population Density Distribution')
axs[2, 1].set_xlabel('Data Index')
axs[2, 1].set_ylabel('Population Density')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
