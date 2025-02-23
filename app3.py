import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os

# Cache the FIPS data loading
@st.cache_data
def load_fips_data():
    fips_df = pd.read_excel("US_FIPS_Codes.xls", skiprows=1).rename(columns={"County Name":"County"})
    fips_df['FIPS County'] = fips_df['FIPS State'].astype(str) + fips_df['FIPS County'].astype(str).str.zfill(3)
    return fips_df.query("State.isin(['Massachusetts','Florida','New York','Georgia','West Virginia'])")

# Cache the county data loading
@st.cache_data
def load_county_data(county):
    return pd.read_parquet(f"county_data/county_{county}")

def deserialize_bytes(geometry_bytes):
    from shapely.wkb import loads
    return loads(geometry_bytes)

def get_lat_long(geometry):
    return geometry.xy[1][0], geometry.xy[0][0]

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

        # Define the path layer
        path_layer = pdk.Layer(
            'PathLayer',
            df,
            get_path='path',
            get_color='color',
            width_scale=10,
            width_min_pixels=2,
            pickable=True,
            auto_highlight=True
        )

        # Set the initial view with higher pitch and adjusted zoom
        view_state = pdk.ViewState(
            latitude=float(lines["lat"].mean()),
            longitude=float(lines["long"].mean()),
            zoom=11,  # Slightly higher zoom
            pitch=0,  # Increased pitch to see terrain better
            bearing=30  # Added bearing for better perspective
        )

        # Create the deck
        deck = pdk.Deck(
            layers=[path_layer],
            initial_view_state=view_state,
            map_style='mapbox://styles/mapbox/satellite-streets-v11',  # Satellite with terrain
            tooltip={'text': 'Path: {path_id}\nVolume: {volume}'}
        )

        return deck

    except Exception as e:
        st.error(f"Error processing county {county}: {str(e)}")
        return None

def main():
    st.set_page_config(layout="wide")  # Use wide mode
    st.title("Traffic Volume Visualization")
    
    # Load FIPS data
    fips_df = load_fips_data()
    
    # Get unique states and sort them
    states = sorted(fips_df['State'].unique())
    
    # Create dropdown for state selection
    selected_state = st.selectbox(
        "Select State",
        options=states,
        index=0
    )
    
    # Filter counties for selected state
    state_counties = fips_df[fips_df['State'] == selected_state]
    
    # Create dropdown for county selection
    selected_county_name = st.selectbox(
        "Select County",
        options=state_counties['County'].tolist(),
        index=0
    )
    
    print(state_counties)
    # Get the FIPS code for the selected county
    selected_county_fips = state_counties[
        state_counties['County'] == selected_county_name
    ]['FIPS County'].iloc[0]
    
    # Generate and display visualization
    if selected_county_fips:
        with st.spinner(f'Loading data for {selected_county_name}, {selected_state}...'):
            deck = generate_viz(selected_county_fips)
            if deck:
                st.pydeck_chart(deck, use_container_width=True)

    
if __name__ == "__main__":
    main()