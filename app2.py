import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import us

# Cache the geocoding results
@st.cache_data
def get_county_from_place(place_name):
    try:
        geolocator = Nominatim(user_agent="traffic_viz_app")
        location = geolocator.geocode(place_name, addressdetails=True)
        
        if location and 'address' in location.raw:
            address = location.raw['address']
            
            # Try to get county information
            county = address.get('county', '')
            state = address.get('state', '')
            
            if county and state:
                # Remove 'County' suffix if present
                county = county.replace(' County', '').replace(' Parish', '')
                
                # Load county FIPS mapping
                fips_df = pd.read_csv('county_fips.csv')  # You'll need to create this mapping
                
                # Find the matching FIPS code
                county_match = fips_df[
                    (fips_df['county'].str.contains(county, case=False)) & 
                    (fips_df['state'].str.contains(state, case=False))
                ]
                
                if not county_match.empty:
                    return county_match.iloc[0]['fips']
                
        return None
        
    except GeocoderTimedOut:
        st.error("Geocoding service timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"Error during geocoding: {str(e)}")
        return None

# Cache only the data loading
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
    
    # Create search box for place input
    place_name = st.text_input(
        "Enter a place name (e.g., 'Seattle, WA' or 'Miami, Florida')",
        help="Enter a city, town, or address to visualize traffic data for that county"
    )
    
    if place_name:
        with st.spinner('Finding county information...'):
            county_fips = get_county_from_place(place_name)
            
            if county_fips:
                with st.spinner(f'Loading data for {place_name}...'):
                    deck = generate_viz(county_fips)
                    if deck:
                        st.pydeck_chart(deck, use_container_width=True)
            else:
                st.error("Could not find county information for the specified place. Please try a different location.")

if __name__ == "__main__":
    main()