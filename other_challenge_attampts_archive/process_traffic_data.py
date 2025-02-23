import pandas as pd
import re

def extract_center_coordinates(linestring):
    coords_str = re.search(r'\((.*?)\)', linestring).group(1)
    coord_pairs = [
        [float(x) for x in pair.strip().split()]
        for pair in coords_str.split(',')
    ]
    
    # Calculate center coordinates
    center_lon = sum(pair[0] for pair in coord_pairs) / len(coord_pairs)
    center_lat = sum(pair[1] for pair in coord_pairs) / len(coord_pairs)
    
    return center_lon, center_lat

def process_traffic_data(file_path):
    """
    Process traffic data CSV and add center coordinates.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Processed dataframe with center coordinates
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Extract center coordinates from geometry column
    centers = df['geometry'].apply(extract_center_coordinates)
    
    df['center_longitude'] = centers.apply(lambda x: x[0])
    df['center_latitude'] = centers.apply(lambda x: x[1])
    
    # df['center_longitude'] = df['center_longitude'].round(6)
    # df['center_latitude'] = df['center_latitude'].round(6)
    
    # df = df.drop(['geometry'], axis=1)
    
    return df

if __name__ == "__main__":
    processed_df = process_traffic_data('traffic_data_sample.csv/traffic_data_sample.csv')
    
    print("\nFirst few rows of processed data:")
    print(processed_df[['segment_name', 'highway', 'center_longitude', 
                       'center_latitude', 'trips_volume']].head())
    processed_df.to_csv('processed_traffic_data.csv', index=False)
    print("\nProcessed data saved to 'processed_traffic_data.csv'")