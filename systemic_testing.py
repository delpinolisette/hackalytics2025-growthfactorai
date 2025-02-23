import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')
from sklearn.neighbors import KNeighborsRegressor



def preprocess_data(df, target_variable="trip_volume"):
    """
    Preprocess the data by encoding categorical variables and scaling numerical variables.

    Parameters:
    df (pandas.DataFrame): Input dataframe
    target_variable (str): Name of the target variable

    Returns:
    tuple: (preprocessed_df, categorical_columns, numerical_columns, encoders)
    """
    # Identify excluded columns
    excluded_cols = [target_variable, "id", "created_at", "updated_at", "vmt"]

    # Get feature columns
    feature_cols = [col for col in df.columns if col not in excluded_cols]

    # Identify categorical and numerical columns
    categorical_cols = df[feature_cols].select_dtypes(
        include=['object', 'category']).columns
    numerical_cols = df[feature_cols].select_dtypes(
        include=['int64', 'float64']).columns

    # Create encoders dictionary
    encoders = {}

    # Process categorical columns with Label Encoding
    df_encoded = df.copy()
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            encoders[col] = LabelEncoder()
            df_encoded[col] = encoders[col].fit_transform(df[col].astype(str))

    return df_encoded, list(categorical_cols), list(numerical_cols), encoders


def systematic_regression(df, target_variable="trip_volume", test_size=0.2, random_state=42,
                          max_features=None, min_features=1, alpha_lasso=1.0, alpha_ridge=1.0, 
                          n_estimators=100):
    """
    System to test regression against trip volume with automatic categorical encoding and feature scaling
    """
    excluded_cols = [target_variable, "id", "created_at", "updated_at", "geom", "day_type", 
                    "day_part", "segment_id", "trips_sample_count", "segment_name", "osm_id", 
                    "trips_sample_count_masked", "trips_volume_masked", "vmt"]
    feature_cols = [col for col in df.columns if col not in excluded_cols]
    
    # Preprocess the data
    df_encoded, categorical_cols, numerical_cols, encoders = preprocess_data(
        df[feature_cols + [target_variable]], target_variable)
    
    # Split into train and test sets
    feature_cols = [col for col in df_encoded.columns if col not in [target_variable]]
    X = df_encoded[feature_cols]
    y = df_encoded[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if max_features is None:
        max_features = len(feature_cols)

    # Initialize scaler
    scaler = StandardScaler()

    # Initialize lists to store results
    results = []

    # Dictionary of models
    models = {
        # 'Linear': LinearRegression(),
        # 'LASSO': Lasso(alpha=alpha_lasso),
        # 'Ridge': Ridge(alpha=alpha_ridge),
        # 'ElasticNet': ElasticNet(alpha=alpha_lasso, l1_ratio=0.5),
        # 'RandomForest': RandomForestRegressor(n_estimators=n_estimators, random_state=42),
        'KNN': KNeighborsRegressor(n_neighbors=2)
    }

    # Test different combinations of features
    # Ensure match_dir is always included
    for n in range(min_features, max_features + 1):
        other_features = [f for f in feature_cols if f != 'match_dir']
        for other_combo in combinations(other_features, n-1):
            feature_combination = ('match_dir',) + other_combo

            X_train_feat = X_train[list(feature_combination)]
            X_test_feat = X_test[list(feature_combination)]
            
            # Scale features
            X_train_scaled = scaler.fit_transform(X_train_feat)
            X_test_scaled = scaler.transform(X_test_feat)
            
            # Convert back to DataFrame to maintain column names
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_feat.columns)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_feat.columns)

            cat_features = [f for f in feature_combination if f in categorical_cols]
            num_features = [f for f in feature_combination if f in numerical_cols]

            model_results = {}
            for model_name, model in models.items():
                try:
                    # Use scaled data for all models
                    model.fit(X_train_scaled, y_train)
                    y_pred_train = model.predict(X_train_scaled)
                    y_pred_test = model.predict(X_test_scaled)

                    train_r2 = r2_score(y_train, y_pred_train)
                    test_r2 = r2_score(y_test, y_pred_test)
                    train_mse = mean_squared_error(y_train, y_pred_train)
                    test_mse = mean_squared_error(y_test, y_pred_test)
                    
                    train_adj_r2 = 1 - (1 - train_r2) * (len(y_train) - 1) / (len(y_train) - len(feature_combination) - 1)
                    test_adj_r2 = 1 - (1 - test_r2) * (len(y_test) - 1) / (len(y_test) - len(feature_combination) - 1)

                    # Store scaled coefficients
                    coef_dict = dict(zip(feature_combination, 
                        model.coef_ if hasattr(model, 'coef_') else
                        model.feature_importances_ if hasattr(model, 'feature_importances_') else
                        [1/len(feature_combination)] * len(feature_combination)))  # Default weights for KNN

                    model_results[model_name] = {
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'train_adj_r2': train_adj_r2,
                        'test_adj_r2': test_adj_r2,
                        'train_mse': train_mse,
                        'test_mse': test_mse,
                        'coefficients': coef_dict,
                        'intercept': model.intercept_ if hasattr(model, 'intercept_') else None
                    }
                    print(model_name)
                    print(model_results)
                except Exception as e:
                    print(f"Warning: {model_name} failed with features {feature_combination}. Error: {str(e)}")
                    continue

            results.append({
                'features': feature_combination,
                'categorical_features': cat_features,
                'numerical_features': num_features,
                'num_features': len(feature_combination),
                **{f'{model_name}_{metric}': values[metric]
                   for model_name, values in model_results.items()
                   for metric in ['train_r2', 'test_r2', 'train_adj_r2', 'test_adj_r2', 'train_mse', 'test_mse']},
                **{f'{model_name}_coefficients': values['coefficients']
                   for model_name, values in model_results.items()},
                **{f'{model_name}_intercept': values['intercept']
                   for model_name, values in model_results.items()}
            })

    return pd.DataFrame(results), encoders, (X_train, X_test, y_train, y_test)


def analyze_top_models(results_df, n_top=5):
    """
    Analyze the top performing models across different regression types.

    Parameters:
    results_df (pandas.DataFrame): Output from systematic_regression function
    n_top (int): Number of top models to analyze (default: 5)

    Returns:
    dict: Analysis of top models including feature importance and comparison
    """

    model_types = ['Linear', 'LASSO', 'Ridge', 'ElasticNet', 'Huber', 'SVR',
                   'KNN', 'DecisionTree', 'RandomForest', 'GradientBoost', 'AdaBoost']
    analysis_results = {}

    for model_type in model_types:
        if f'{model_type}_test_adj_r2' not in results_df.columns:
            continue

        # Sort by test adjusted R-squared for this model type
        top_models = results_df.sort_values(
            f'{model_type}_test_adj_r2', ascending=False).head(n_top)

        # Analyze feature frequency in top models
        feature_frequency = {}
        for _, row in top_models.iterrows():
            for feature in row['features']:
                if feature not in feature_frequency:
                    feature_frequency[feature] = 0
                feature_frequency[feature] += 1

        # Analyze coefficient/importance stability
        coefficient_stats = {}
        for feature in feature_frequency.keys():
            coefficients = []
            for _, row in top_models.iterrows():
                coef_dict = row[f'{model_type}_coefficients']
                if feature in coef_dict:
                    coefficients.append(coef_dict[feature])

            if coefficients:
                coefficient_stats[feature] = {
                    'mean': np.mean(coefficients),
                    'std': np.std(coefficients),
                    'frequency': feature_frequency[feature]
                }

        analysis_results[model_type] = {
            'top_models': top_models,
            'feature_frequency': feature_frequency,
            'coefficient_stats': coefficient_stats
        }

    return analysis_results


def print_analysis_summary(analysis_results, n_top=5):
    """
    Print a summary of the analysis results.

    Parameters:
    analysis_results (dict): Output from analyze_top_models function
    n_top (int): Number of top models included in analysis
    """
    model_types = ['Linear', 'LASSO', 'Ridge', 'ElasticNet', 'Huber', 'SVR',
                   'KNN', 'DecisionTree', 'RandomForest', 'GradientBoost', 'AdaBoost']

    for model_type in model_types:
        if model_type not in analysis_results:
            continue

        print(f"\n=== {model_type} Regression Analysis ===")
        results = analysis_results[model_type]

        print(f"\nTop {n_top} Models (sorted by test adjusted R²):")
        metrics = ['features', f'{model_type}_train_adj_r2', f'{model_type}_test_adj_r2',
                  f'{model_type}_train_mse', f'{model_type}_test_mse']
        top_models_summary = results['top_models'][metrics]
        print(top_models_summary)

        print("\nFeature Frequency in Top Models:")
        for feature, freq in results['feature_frequency'].items():
            print(f"{feature}: {freq} times")

        print("\nFeature Importance/Coefficient Statistics:")
        for feature, stats in results['coefficient_stats'].items():
            print(f"\n{feature}:")
            print(f"Mean coefficient/importance: {stats['mean']:.4f}")
            print(f"Standard deviation: {stats['std']:.4f}")
            print(f"Frequency in top models: {stats['frequency']}")


def print_encoding_summary(df, encoders):
    """
    Print a summary of the categorical encoding.

    Parameters:
    df (pandas.DataFrame): Original dataframe
    encoders (dict): Dictionary of label encoders
    """
    print("\n=== Categorical Encoding Summary ===")
    for col, encoder in encoders.items():
        print(f"\nColumn: {col}")
        mapping = dict(
            zip(encoder.classes_, encoder.transform(encoder.classes_)))
        print("Value mappings:")
        for original, encoded in mapping.items():
            print(f"  {original} -> {encoded}")


def log_model_coefficients(analysis_results, output_file='model_coefficients.csv'):
    """
    Log model coefficients and feature importances to a CSV file.

    Parameters:
    analysis_results (dict): Output from analyze_top_models function
    output_file (str): Name of the output CSV file
    """
    all_records = []

    for model_type, results in analysis_results.items():
        top_models = results['top_models']

        for _, row in top_models.iterrows():
            coef_dict = row[f'{model_type}_coefficients']
            features = row['features']
            train_adj_r2 = row[f'{model_type}_train_adj_r2']
            test_adj_r2 = row[f'{model_type}_test_adj_r2']
            train_mse = row[f'{model_type}_train_mse']
            test_mse = row[f'{model_type}_test_mse']

            for feature, coef in coef_dict.items():
                record = {
                    'model_type': model_type,
                    'feature': feature,
                    'coefficient': coef,
                    'feature_set': str(features),
                    'train_adj_r2': train_adj_r2,
                    'test_adj_r2': test_adj_r2, 
                    'train_mse': train_mse,
                    'test_mse': test_mse
                }
                all_records.append(record)

    # Create DataFrame and save to CSV
    coef_df = pd.DataFrame(all_records)
    coef_df.to_csv(output_file, index=False)
    print(
        f"\nModel coefficients and feature importances have been saved to {output_file}")


# Load your dataset
# breakpoint()
df = pd.read_csv('processed_traffic_data.csv')

# Run systematic regression with all models
results, encoders, (X_train, X_test, y_train, y_test) = systematic_regression(df,
                                          target_variable='trips_volume',
                                          max_features=3,
                                          min_features=1,
                                          alpha_lasso=1.0,
                                          alpha_ridge=1.0,
                                          n_estimators=100)
# print_encoding_summary(df, encoders)


# Analyze top models
analysis = analyze_top_models(results, n_top=5)

# Print comprehensive analysis
print_analysis_summary(analysis, n_top=5)

log_model_coefficients(analysis, 'model_coefficients.csv')