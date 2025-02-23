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

@st.cache_data
def load_county_data(county):
    return pd.read_parquet(f"county_data/county_{county}")

def deserialize_bytes(geometry_bytes):
    from shapely.wkb import loads
    return loads(geometry_bytes)

def get_lat_long(geometry):
    return geometry.xy[1][0], geometry.xy[0][0]

def generate_viz(county, viz_type='path', marker_lat=None, marker_lon=None):
    try:
        # Load only necessary columns
        lines = load_county_data(county)[['geometry', 'osm_id', 'trips_volu']]
        
        # Process geometry in vectorized operations
        lines['geometry'] = lines['geometry'].apply(deserialize_bytes)
        lines["lat"], lines["long"] = zip(*lines['geometry'].apply(get_lat_long))
        
        # Calculate log-transformed trips for better visualization
        log_trips = np.log1p(lines["trips_volu"])
        norm = Normalize(vmin=log_trips.min(), vmax=log_trips.max())
        colormap = ScalarMappable(norm=norm, cmap="viridis")
        colors = (colormap.to_rgba(log_trips)[:, :3] * 255).astype(int).tolist()

        # Set initial view state
        view_state = pdk.ViewState(
            latitude=float(lines["lat"].mean()),
            longitude=float(lines["long"].mean()),
            zoom=11,
            pitch=0,
            bearing=30
        )

        layers = []

        if viz_type == 'path':
            # Create line data for path visualization
            line_data = [[[float(x), float(y)] for x, y in zip(*geom.xy)] 
                        for geom in lines['geometry']]

            df = pd.DataFrame({
                'path_id': lines["osm_id"],
                'path': line_data,
                'color': colors,
                'volume': lines["trips_volu"]
            })

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
            
            layers.append(path_layer)
            tooltip = {'text': 'Path: {path_id}\nVolume: {volume}'}

        else:  # heatmap
            # Create point data for heatmap
            df = pd.DataFrame({
                'latitude': lines["lat"],
                'longitude': lines["long"],
                'weight': lines["trips_volu"]
            })

            heatmap_layer = pdk.Layer(
                'HeatmapLayer',
                df,
                opacity=0.8,
                get_position=['longitude', 'latitude'],
                get_weight='weight',
                aggregation='"SUM"',
                threshold=0.05,
                radius_pixels=30,
                color_range=[
                    [65, 182, 196],
                    [127, 205, 187],
                    [199, 233, 180],
                    [237, 248, 177],
                    [255, 255, 204],
                    [255, 237, 160],
                    [254, 217, 118],
                    [254, 178, 76],
                    [253, 141, 60],
                    [252, 78, 42],
                    [227, 26, 28],
                    [189, 0, 38]
                ]
            )
            
            layers.append(heatmap_layer)
            tooltip = None

        # Add marker layer if coordinates are provided
        if marker_lat is not None and marker_lon is not None:
            marker_data = pd.DataFrame({
                'latitude': [marker_lat],
                'longitude': [marker_lon]
            })
            
            marker_layer = pdk.Layer(
                'ScatterplotLayer',
                marker_data,
                get_position=['longitude', 'latitude'],
                get_color=[255, 0, 0, 200],  # Red marker with some transparency
                get_radius=20,
                pickable=True,
                radiusScale=6,
                radiusMinPixels=5,
                radiusMaxPixels=20
            )
            
            layers.append(marker_layer)

        # Create the deck
        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            map_style='mapbox://styles/mapbox/satellite-streets-v11',
            tooltip=tooltip
        )

        return deck

    except Exception as e:
        st.error(f"Error processing county {county}: {str(e)}")
        return None

def main():
    st.set_page_config(layout="wide")
    st.title("Traffic Volume Visualization")
    
    # Load FIPS data
    fips_df = load_fips_data()
    states = sorted(fips_df['State'].unique())
    
    # Create columns for all controls in a single row
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    
    with col1:
        selected_state = st.selectbox(
            "Select State",
            options=states,
            index=0
        )
    
    # Filter counties for selected state
    state_counties = fips_df[fips_df['State'] == selected_state]
    
    with col2:
        selected_county_name = st.selectbox(
            "Select County",
            options=state_counties['County'].tolist(),
            index=0
        )
    
    with col3:
        viz_type = st.selectbox(
            "Visualization Type",
            options=['Path View', 'Heatmap'],
            index=0
        )
    
    with col4:
        marker_lat = st.text_input("Latitude", value="", help="Enter latitude for marker (optional)")
    
    with col5:
        marker_lon = st.text_input("Longitude", value="", help="Enter longitude for marker (optional)")
    
    # Convert coordinates to float if provided
    try:
        marker_lat = float(marker_lat) if marker_lat else None
        marker_lon = float(marker_lon) if marker_lon else None
    except ValueError:
        st.error("Please enter valid numerical coordinates")
        marker_lat, marker_lon = None, None
    
    # Get the FIPS code for the selected county
    selected_county_fips = state_counties[
        state_counties['County'] == selected_county_name
    ]['FIPS County'].iloc[0]
    
    # Generate and display visualization
    if selected_county_fips:
        with st.spinner(f'Loading data for {selected_county_name}, {selected_state}...'):
            deck = generate_viz(
                selected_county_fips, 
                'path' if viz_type == 'Path View' else 'heatmap',
                marker_lat,
                marker_lon
            )
            if deck:
                st.pydeck_chart(deck, use_container_width=True)

if __name__ == "__main__":
    main()