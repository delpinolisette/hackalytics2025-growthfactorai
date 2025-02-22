import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

def systematic_regression(df, target_variable="trip_volume", max_features=None, min_features=1):
    """
    Perform systematic regression analysis testing different feature combinations.
    
    Parameters:
    df (pandas.DataFrame): Input dataframe
    target_variable (str): Name of the target variable
    max_features (int): Maximum number of features to combine (default: None = all features)
    min_features (int): Minimum number of features to combine (default: 1)
    
    Returns:
    pandas.DataFrame: Results of all regression combinations, sorted by R-squared
    """
    
    # Get feature columns (excluding target variable) and id columns
    feature_cols = [col for col in df.columns if col not in [target_variable, "id", "created_at", "updated_at"]]

    # TODO: create the center of the lat and longitude strings 
    
    if max_features is None:
        max_features = len(feature_cols)
    
    # Initialize lists to store results
    results = []
    
    # Test different combinations of features
    for n in range(min_features, max_features + 1):
        for feature_combination in combinations(feature_cols, n):
            # Create X (features) and y (target)
            breakpoint()
            X = df[list(feature_combination)]
            y = df[target_variable]
            
            # Fit regression model
            model = LinearRegression()
            model.fit(X, y)
            
            # Make predictions
            y_pred = model.predict(X)
            
            # Calculate metrics
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - len(feature_combination) - 1)
            
            # Get coefficients
            coef_dict = dict(zip(feature_combination, model.coef_))
            
            # Store results
            results.append({
                'features': feature_combination,
                'num_features': len(feature_combination),
                'r2_score': r2,
                'adjusted_r2': adj_r2,
                'mse': mse,
                'coefficients': coef_dict,
                'intercept': model.intercept_
            })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by adjusted R-squared (descending)
    results_df = results_df.sort_values('adjusted_r2', ascending=False)
    
    return results_df

def analyze_top_models(results_df, n_top=5):
    """
    Analyze the top performing regression models.
    
    Parameters:
    results_df (pandas.DataFrame): Output from systematic_regression function
    n_top (int): Number of top models to analyze (default: 5)
    
    Returns:
    dict: Analysis of top models including feature importance and comparison
    """
    
    top_models = results_df.head(n_top)
    
    # Analyze feature frequency in top models
    feature_frequency = {}
    for _, row in top_models.iterrows():
        for feature in row['features']:
            if feature not in feature_frequency:
                feature_frequency[feature] = 0
            feature_frequency[feature] += 1
    
    # Analyze coefficient stability
    coefficient_stats = {}
    for feature in feature_frequency.keys():
        coefficients = []
        for _, row in top_models.iterrows():
            if feature in row['coefficients']:
                coefficients.append(row['coefficients'][feature])
        
        if coefficients:
            coefficient_stats[feature] = {
                'mean': np.mean(coefficients),
                'std': np.std(coefficients),
                'frequency': feature_frequency[feature]
            }
    
    return {
        'top_models': top_models,
        'feature_frequency': feature_frequency,
        'coefficient_stats': coefficient_stats
    }

# driver code 
df = pd.read_csv('traffic_data_sample.csv/traffic_data_sample.csv')
results = systematic_regression(df, 
                             target_variable='trips_volume',
                             max_features=None,  # Optional: limit max features
                             min_features=1)  # Optional: set minimum features


# Example usage:
"""
# Load your dataset
df = pd.read_csv('your_dataset.csv')

# Run systematic regression
results = systematic_regression(df, 
                             target_variable='target_column',
                             max_features=4,  # Optional: limit max features
                             min_features=1)  # Optional: set minimum features

# Analyze top models
analysis = analyze_top_models(results, n_top=5)

# Print top models
print("\nTop 5 Models:")
print(analysis['top_models'][['features', 'adjusted_r2', 'mse']])

# Print feature importance analysis
print("\nFeature Frequency in Top Models:")
for feature, freq in analysis['feature_frequency'].items():
    print(f"{feature}: {freq} times")

# Print coefficient stability analysis
print("\nCoefficient Statistics:")
for feature, stats in analysis['coefficient_stats'].items():
    print(f"\n{feature}:")
    print(f"Mean coefficient: {stats['mean']:.4f}")
    print(f"Coefficient std: {stats['std']:.4f}")
    print(f"Frequency in top models: {stats['frequency']}")
"""