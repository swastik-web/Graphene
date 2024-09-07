import pandas as pd
import numpy as np

# Generate mock satellite data (coarse resolution)
np.random.seed(42)
n_samples = 1000

satellite_data = pd.DataFrame({
    'latitude': np.random.uniform(low=35.0, high=40.0, size=n_samples),
    'longitude': np.random.uniform(low=-120.0, high=-115.0, size=n_samples),
    'date': pd.date_range(start='2024-01-01', periods=n_samples, freq='H').strftime('%Y-%m-%d'),
    'satellite_no2': np.random.uniform(low=0, high=100, size=n_samples),
    'temperature': np.random.uniform(low=280, high=320, size=n_samples),  # In Kelvin
    'humidity': np.random.uniform(low=30, high=90, size=n_samples),  # In percentage
    'population_density': np.random.uniform(low=50, high=5000, size=n_samples),
    'proximity_to_roads': np.random.uniform(low=0, high=10, size=n_samples)  # In km
})

# Save to CSV
satellite_data.to_csv('Downscaling of Satellite based air quality map using AI/mock_satellite_data.csv', index=False)

# Generate mock ground truth data (fine resolution)
ground_data = pd.DataFrame({
    'latitude': satellite_data['latitude'],  # Assume same locations as satellite data
    'longitude': satellite_data['longitude'],
    'date': satellite_data['date'],
    'ground_no2': satellite_data['satellite_no2'] * np.random.uniform(low=0.7, high=1.3, size=n_samples)  # Add some noise
})

# Save to CSV
ground_data.to_csv('Downscaling of Satellite based air quality map using AI/mock_ground_truth.csv', index=False)

# Generate mock independent data for validation
independent_data = pd.DataFrame({
    'latitude': np.random.uniform(low=35.0, high=40.0, size=n_samples),
    'longitude': np.random.uniform(low=-120.0, high=-115.0, size=n_samples),
    'date': pd.date_range(start='2024-02-01', periods=n_samples, freq='H').strftime('%Y-%m-%d'),
    'satellite_no2': np.random.uniform(low=0, high=100, size=n_samples),
    'temperature': np.random.uniform(low=280, high=320, size=n_samples),
    'humidity': np.random.uniform(low=30, high=90, size=n_samples),
    'population_density': np.random.uniform(low=50, high=5000, size=n_samples),
    'proximity_to_roads': np.random.uniform(low=0, high=10, size=n_samples),
    'ground_no2': np.random.uniform(low=0, high=100, size=n_samples)  # Simulate ground-level data
})

# Save to CSV
independent_data.to_csv('Downscaling of Satellite based air quality map using AI/mock_independent_data.csv', index=False)
