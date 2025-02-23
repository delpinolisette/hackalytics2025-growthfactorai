import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os

# Cache only the data loading
@st.cache_data
def load_county_data(county):
    return pd.read_parquet(f"county_data/county_{county}")

def deserialize_bytes(geometry_bytes):
    from shapely.wkb import loads
    return loads(geometry_bytes)

def get_lat_long(geometry):
    return geometry.xy[1][0], geometry.xy[0][0]

# Removed caching from visualization generation
def generate_viz(county):
    try:
        # Load only necessary columns
        lines = load_county_data(county)[['geometry', 'osm_id', 'trips_volu']]
        
        # Process geometry in vectorized operations where possible
        lines['geometry'] = lines['geometry'].apply(deserialize_bytes)
        lines["lat"], lines["long"] = zip(*lines['geometry'].apply(get_lat_long))
        
        # Vectorize color calculation
        log_trips = np.log1p(lines["trips_volu"])
        norm = Normalize(vmin=log_trips.min(), vmax=log_trips.max())
        colormap = ScalarMappable(norm=norm, cmap="viridis")
        colors = (colormap.to_rgba(log_trips)[:, :3] * 255).astype(int).tolist()
        
        # Create line data more efficiently
        line_data = [[[float(x), float(y)] for x, y in zip(*geom.xy)] 
                    for geom in lines['geometry']]

        # Create DataFrame for the paths
        df = pd.DataFrame({
            'path_id': lines["osm_id"],
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
    st.set_page_config(layout="wide")  # Use wide mode
    st.title("Traffic Volume Visualization")
    
    # Cache the county list
    @st.cache_data
    def get_counties():
        county_files = [f for f in os.listdir("county_data") if f.startswith("county_")]
        return [int(f.split("_")[1]) for f in county_files]
    
    counties = get_counties()
    
    # Create dropdown for county selection
    selected_county = st.selectbox(
        "Select County (FIPS code)",
        options=counties,
        format_func=lambda x: f"County {x}"
    )
    
    # Generate and display visualization
    if selected_county:
        with st.spinner(f'Loading data for county {selected_county}...'):
            deck = generate_viz(selected_county)
            if deck:
                st.pydeck_chart(deck, use_container_width=True)

if __name__ == "__main__":
    main()