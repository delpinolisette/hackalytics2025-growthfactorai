import pandas as pd
df_lst = []
for county in tqdm(set(gdf['county_fip'])):
    county_df = gdf.query(f"county_fip == {county}")
    #county_df['geometry'] = county_df['geometry'].apply(deserialize_bytes)
    county_df["lat"], county_df["long"] = zip(*county_df['geometry'].apply(get_lat_long))
    df_lst.append(county_df[['county_fip','state_code','highway','segment_na','lat','long','geometry', 'osm_id', 'trips_volu']])
traffic_data_condensed = pd.concat(df_lst)
traffic_data_condensed.to_parquet("traffic_data_condensed.parquet")