- [Final Idea: Geospatially-Motivated Scoring and Visualization](#final-idea-geospatially-motivated-scoring-and-visualization)
  - [Note on Final Product](#note-on-final-product)
  - [Data Preprocessing and Ingestion](#data-preprocessing-and-ingestion)
  - [Initial EDA - Trip Volume Distribution](#initial-eda---trip-volume-distribution)
  - [Algorithm](#algorithm)
    - [KD Tree for Range Search](#kd-tree-for-range-search)
    - [Notes on `match_id`](#notes-on-match_id)
    - [Note on identical rows except `trip_volume`](#note-on-identical-rows-except-trip_volume)
    - [Zooming In on an Example:](#zooming-in-on-an-example)
  - [The Impressions Calculation: Crux of the Algorithm](#the-impressions-calculation-crux-of-the-algorithm)
    - [The "County Percentile" Heuristic](#the-county-percentile-heuristic)
      - [Use the pre-made `county` Partitions!](#use-the-pre-made-county-partitions)
- [Application](#application)
- [Other Methods (Which Did Not Work)](#other-methods-which-did-not-work)
  - [Systematic Regression Analysis](#systematic-regression-analysis)
- [Future Directions and Ideas:](#future-directions-and-ideas)
- [Testing/Validation of our Approach](#testingvalidation-of-our-approach)

# Final Idea: Geospatially-Motivated Scoring and Visualization

## Note on Final Product

Please note that to arrive to our final product and algorithm, we had to throw out plenty of ideas and hundreds of lines of code. 

We settled on the visualization driven method, which aligned closely with visual intuition. The full analysis details, examples, and observations leading to this method are well-documented on the Jupyter notebook. **Please refer to that notebook for more discussion.**

We also created a deployed Streamlit application, which allows the user to input a pair of coordinates; returns the score, and visualizes the trip volumes on a map, color-graded (by Kernel Density Estimation of nearby segments) intensities for usability/explorability. You can watch or experience the application demo and read more about it in our [application section](#application).

## Data Preprocessing and Ingestion

We first ingested the geopandas version of the data. Our first improvement on the existing data to improve performance was to partition the dataset into state and counties.

This partitioning proved to be crucial for the performance during data querying during exploration, data querying on the deployed Streamlit application, and the algorithm itself (which visualizes at the county level and uses the county for the impression percentile scoring).

Our next processing step, after discussing some simplifying assumptions with Raj and Tanner, was to remove pedestrian, cycleways, and footpaths from the data. In the OSM HIGHWAY column, a handful of rows were classified as these. This analysis was restricted to impressions from car observers and traffic only.

## Initial EDA - Trip Volume Distribution

In the initial EDA we found that trip volumes, a component of the impressions target variable, has a heavy right tail, with some of the summary statistics being a mean of 5,930, a standard deviation of 9,331, a median of 3,012; and a max value of 232,019. The distribution of trip volume is in the chart below:

<!-- ![](../assets/2025-02-22-23-47-18.png) -->
<img src="../assets/dist_trip_vol.png" width="80%" alt="description">

## Algorithm

Our algorithm for determining impressions involves a kd-tree for range search to obtain nearby segments. Once the nearby segments are obtained, we then select segments with correct orientation, and aggregate their trip volumes to obtain an aggregate trip volumne score.  We then use calibrated thresholds based on county data to obtain an impression score between 0 and 1.

### KD Tree for Range Search

The first step of the algorithm is to retrieve all points contained within a certain radius of an input point (store). We use the haversine as our distance metric.  The KD tree reduces the search time from linear to logarithmic complexity.  To further optimize performance, we restrict the search to only points within the same county.  

### Notes on `match_id`

After checking a street we know well in Boston, we noticed that for a segment id, the one that has match_dir = 1 and the one with match_dir = 2 add up to the observation with match_dir = 3.

So, when match_dir = 1, that `trips_volume` corresponds to the "correct"/"closest" side of the segment to the geometry. Thus, we have to consider `match_dir = 1` when computing impressions. This is a massively simplifying assumption as drivers can certainly look at other sides of the road while waiting at a red light, but are more likely to notice businesses on their side of the road. It will also be easier to pull into same road side businesses for most of the US, barring highways with no nearby exit to the adjacent side of the road.

### Note on identical rows except `trip_volume`

Since we noticed several instances where rows seemed identical minus the trip volume, which was interesting, we wanted to take those average of those volumes and combined them into one observation.

After an unpacking of the geom data, we realized we could not dedupe as these were distinct geometries tied to distinct trip volumes. However, these distinct geometries were sometimes tied to the same Open Street Map id, which was curious, one such example occuring over 200 times.

### Zooming In on an Example:

For example, when we zoom in on this provided store location (black dot) in Harvard Square, and filter for `match_id = 1`, we get segments that are on the correct side of the street to get impressions facing the storefront. In real life, those segments do have a direct line of sight to that store location, verifying our visual intuition from the map visualization:

<!-- ![](../assets/2025-02-23-01-38-37.png) -->
<img src="../assets/2025-02-23-01-38-37.png" width="80%" alt="description">

## The Impressions Calculation: Crux of the Algorithm

We can answer the **impressions** question posed by the team by taking the mean of trip volumes for all nearest neighbors (computed by our KD Tree based function), for a provided pair of coordinates or geometries input, with some careful constraints. 

Note: When we aggregate the trip volumes for the segments in the given radius, we do a simple average, but this could be improved with a Parzen Window/density estimation.

```python
neighbors_df[neighbors_df['id'].isin(set(county_data_neighbors['id']))]['trips_volu'].mean()
```

Once we've aggregated the nearby trip volumes, we compute the impression score. 

We would like to normalize the impression score to be between 0 and 1. To perform the calibration of this score, we compute a set of `thresholds` based on the distribution of trip volumes in the county. The reason that this calibration is done at the county level is because it seems reasonable that a client may want to find a piece of property nearby at the local level, rather than a national search. See the below section for further discussion on this heuristic. The **$k$-th percentile** of trip volumes corresponds to the **$k$-th impression score**.

<img src="../assets/impression_score_curve.png" width="80%" alt="description">

Note: There are other ways to calibrate the impression score. For example, instead of constraining the trip volumes, we could use the constraint than an **equal proportion of segments** have each impression score.  

### The "County Percentile" Heuristic

Counties are a natural demographic and political divisions, as well as useful data partitions for performance. If someone is looking for a future shop location, they will likely want to stay within a certain county.

Also, oftentimes, financial policies and business incentives set at the county level. For example, Cobb County has a small business incentive program, which will encourage clients to plan their store within its confines.

#### Use the pre-made `county` Partitions!

Since we had (in a sense) taken care of "wrong side of the road" impressions with `match_dir = 1`, and they are a useful heuristic as a "percentile background", we can also use the county partitions that already existed in our data. 


# Application

The application has user settings of state, county, and optional latitude and longitude pair to compute an impression score for. Since trip volume is a critical part of the impression calculation, the focus of this application is visualizing trip volume.  

There are two views in the application.  The first view colors the segments by trip volume, with the color scale on the log scale, with a gradient color scheme. The trip volume of segments can easily be compared in this first view. The second view uses kernel density estimation to color areas around the segments.  


<img src="../assets/kernel_density_heatmap.png" width="80%" alt="description">
<img src="../assets/kernel_density_heatmap2.png" width="80%" alt="description">

As you can see, the application returns a score for a set of latitudes and longitudes, and also provides an interactive visualization with satellite image overlays for clients.  

As a bonus, our app also works great on mobile! Clients would be able to use it on the go. 

# Future Directions and Ideas:

Parzen Window/density estimation
- We would like to refactor the aggregation of trip volumes under our given constraints to use Parzen Window/density estimation, rather than a simple average.


Graphical Models and Graph Databases:

- Given more time, we would have liked to explore segment connections and relationships with graph relationships. Perhaps tools such as graph neural networks or community detection algorithms may be useful in this context.

# Testing/Validation of our Approach

We tested a few local (to us) locations in Boston with extremely high traffic and impressions, and our algorithm and application yielded a high score for all of these. 

We also noticed that our method picked up on **subtle traffic patterns and brought those into our impressions score**. A location in an area that had a high potential impressions score scored lower than an adjacent storefront which was on an intersection with two streets facing it, which allowed for more eyes on the store location from car traffic. 


