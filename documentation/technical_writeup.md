- [Systematic Regression Analysis to get intution](#systematic-regression-analysis-to-get-intution)
- [Idea 2: Try this with ensemble method](#idea-2-try-this-with-ensemble-method)
- [nearest neighbors method](#nearest-neighbors-method)
- [convolution method](#convolution-method)
- [future directions:](#future-directions)
- [Idea 3: Graphical Neural Net](#idea-3-graphical-neural-net)
- [Final Idea: Geospatial function](#final-idea-geospatial-function)
  - [Algorithm](#algorithm)
    - [Point 1:](#point-1)
    - [Point 2:](#point-2)


# Systematic Regression Analysis to get intution

We originally brainstormed a variety of methods, attacking the **impressions** problem by getting intuition both through visual methods and through statistical methods. 

To start, we ran a systematic regression analysis, where 

target = volume = trips_volume

this one is not perfect as the data is masked... 
removed a few confusing columns. 

also i unrolled the latitude and longitude and we just got the center of the data, since using the rolled up version as a feature would not work

for categorical variables we encoded them automatically 

and tried a variety of regression methods 

measure coefficients and feature importance 

simplifying assumption: take the center of latitude and longitude groups 

we systematically analyze any number and combination of features against the target, which is trips volunme 


these are the results:

* Performs systematic regression analysis on traffic data using multiple models (Linear, LASSO, Ridge, etc.)
  * Linear Regression - Basic linear model with no regularization
  * LASSO - L1 regularization, can eliminate features
  * Ridge - L2 regularization, shrinks feature coefficients
  * ElasticNet - Combines L1 and L2 regularization
  * HuberRegressor - Robust to outliers
  * KNN - Predicts based on nearest neighbors
  * DecisionTree - Tree-based predictions
  * RandomForest - Ensemble of trees
  * GradientBoosting - Sequential tree building
  * AdaBoost - Focuses on hard-to-predict cases
* Preprocesses data by encoding categorical variables and scaling numerical features
* Tests different combinations of features to find best predictors of trip volume
* Analyzes top performing models and their feature importance
* Creates detailed reports including model coefficients, R-squared scores, and MSE
* Saves model coefficients and analysis results to CSV files
* Uses various validation metrics to evaluate model performance
* Includes functions to print summaries of the analysis and encoding mappings
* Avoids certain columns (like IDs and timestamps) that aren't relevant for prediction
* Handles both categorical and numerical features appropriately for different model types


# Idea 2: Try this with ensemble method

follows from the regression 

# nearest neighbors method 

inspired by the visualization by bens 
nn img 

# convolution method

# future directions:

kernet heat map 

# Idea 3: Graphical Neural Net


# Final Idea: Geospatial function

- ingest.py function to ingest 
- our first speedup was reading geopandas file and paritioning it into each state and county to make the visualization easier to search for its respective 
- our next processing step was to 

## Algorithm 

### Point 1: 

- convolution with **each** individual point

$$f(\vec{s}, \vec{d}) \in \mathbb{R}^1$$ 

### Point 2: 
