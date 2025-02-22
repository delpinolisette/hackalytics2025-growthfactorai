import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os

def deserialize_bytes(geometry_bytes):
    from shapely.wkb import loads
    return loads(geometry_bytes)

def get_lat_long(geometry):
    return geometry.xy[1][0], geometry.xy[0][0]

def generate_viz(county):
    try:
        lines = pd.read_parquet(f"county_data/county_{county}")
        lines['geometry'] = lines['geometry'].apply(deserialize_bytes)
        lines["lat"], lines["long"] = zip(*lines['geometry'].apply(get_lat_long))
        
        # List to hold the line data for Pydeck
        line_data = []

        # Create the line data from the geometries
        for line in lines['geometry']:
            x, y = line.xy
            x, y = np.array(x), np.array(y)
            
            # Prepare the coordinates for Pydeck
            line_coords = [[a, b] for a, b in zip(x, y)] 
            line_data.append(line_coords)

        colors = []
        lines["log_trips_volu"] = np.log(lines["trips_volu"]+1)
        for trip_volume in lines["log_trips_volu"]:
            # Normalize the 'trips_volu' to a color range
            norm = Normalize(vmin=lines['log_trips_volu'].min(), vmax=lines['log_trips_volu'].max())
            colormap = ScalarMappable(norm=norm, cmap="viridis")
            color = colormap.to_rgba(trip_volume)[:3]
            colors.append([int(c * 255) for c in color])

        # Create DataFrame for the paths
        df = pd.DataFrame({
            'path_id': list(lines["osm_id"]),
            'path': line_data,
            'color': colors,
            'volume': lines["trips_volu"]
        })

        # Define the layer
        layer = pdk.Layer(
            'PathLayer',
            df,
            get_path='path',
            get_color='color',
            width_scale=20,
            width_min_pixels=2,
            pickable=True,
            auto_highlight=True
        )

        # Set the initial view
        view_state = pdk.ViewState(
            latitude=float(lines["lat"].mean()),
            longitude=float(lines["long"].mean()),
            zoom=10,
            pitch=45,
            bearing=0
        )

        # Create the deck
        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={'text': 'Path: {path_id}\nVolume: {volume}'}
        )

        return deck

    except Exception as e:
        st.error(f"Error processing county {county}: {str(e)}")
        return None

def main():
    st.title("Traffic Volume Visualization")
    
    # Get available county files
    county_files = [f for f in os.listdir("county_data") if f.startswith("county_")]
    print(county_files)
    counties = [int(f.split("_")[1]) for f in county_files]
    
    # Create dropdown for county selection
    selected_county = st.selectbox(
        "Select County (FIPS code)",
        options=counties,
        format_func=lambda x: f"County {x}"
    )
    
    # Generate and display visualization
    if selected_county:
        st.write(f"Displaying traffic data for county {selected_county}")
        deck = generate_viz(selected_county)
        if deck:
            st.pydeck_chart(deck)

if __name__ == "__main__":
    main()