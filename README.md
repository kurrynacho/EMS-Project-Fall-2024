# Predicting Call Volumes for the Emergency Medical Services (EMS)
### Erdos Institute data science project, Fall 2024

## Authors
- [Darius Alizadeh](https://github.com/dariusdalizadeh)
- [Karina Cho](https://github.com/kurrynacho)
- [Jessica Liu](https://github.com/jessicaliu11)
- [Jonathan Miller](https://github.com/mllrjo)

## Summary

Our broad goal is to predict the volume of different types of Emergency Medical Service (EMS) calls within a certain time interval with time series methods and to identify features that correlate with the volume of EMS calls. 


## Dataset

We work with a public data set provided by the National Emergency Medical Services Information System (NEMSIS), a product of the National Highway Safety Administration's EMS Office that sets the standard for EMS agencies to collect and report data. It consolidates EMS data from 54 participating states and territories into a national database. We have obtained public datasets from 2018 through 2023 that are purged of identifying information to comply with privacy law. A single EMS event is represented by (possibly multiple) EMS activations when multiple units respond to the scene.


## Exploratory Data Analysis

(Summarize interesting EDA)

## Data Engineering and Preprocessing

(Summarize work done on data engineering in SQL)

![]() <img src="images/call_volumes0JTMM.png" width=95%>

Plotting the time series data for the call volume, we see that the data has a lot of instability. In particular, we see increases in call volumes between years, which is likely due to more EMS centers reporting data over time. To account for this, we normalize the call volume for each year to obtain a more stationary data set, which is necessary in order to fit SARIMA models.

![]() <img src="images/scaledcallvol.png" width=95%>

## Modeling
### SARIMA Models

Based on the EDA, there seems to be a seasonal component to the data, which can be captured by a SARIMA model. Since each county appeared to have different behaviour, we decided to train models on single counties. Here, we describe the process for producing a model for a single county.

We perform analysis on the autocorrelation and partial autocorellation functions in order to deduce reasonable parameters for our SARIMA(p,d,q)(P,D,Q)[m] model. 

<p float="left">
  <img src="images/autocorr0JTMM.png" width="48%" />
  <img src="images/partialautocorr0JTMM.png" width="48%" /> 
</p>

- The ACF and PACF both have significant lags at multiples of 7, which suggests that we have a seasonality of m=7.

Using the Akaike information Criterion (AIC), which estimates prediction error and allows us to weigh the relative quality of certain models, we perform our model selection by doing a stepwise search among parameters.  We select three models with low AIC values and compare these models against a baseline naive model using cross validation.

![]() <img src="images/modelcomparison.png" width=95%>

Here are the mean squared errors for these four models:
| Model   | MSE |
| -------- | ------- |
| Naive model                | 0.05129    |
| SARIMA(0,1,1)(3,0,1)[7]    | 0.03308    |
| SARIMA(1,1,1)(2,0,1)[7]    | 0.03244    |
| SARIMA(0,1,2)(2,0,1)[7]    | 0.03240    |

The naive model had the largest MSE, and all three selected SARIMA models cut the error by over a third. The SARIMA(1,2,2)(2,0,1)[7] performed the best.
### State Space Models

### Combining State Space and SARIMA Models
### Neural Network

## Python Modules
# Documentation

## Modules

### **`loading_data.py`**
This module contains Python scripts to process large SAS datasets efficiently. The pipeline includes reading SAS files, converting them to Parquet format, filtering and merging datasets, and organizing data by state. The implementation uses **Pandas** for smaller tasks and **Dask** for distributed processing of large datasets.

#### Features

1. **SAS to Parquet Conversion (`sas_to_parquet`)**:
   - Reads large SAS files in chunks and converts them to Parquet files for efficient storage and processing.
   - Creates a designated folder to store Parquet files if it doesn't already exist.

2. **Parquet File Loading (`parquet_to_df`)**:
   - Combines Parquet files into a Dask DataFrame for scalable data analysis.

3. **Filtered DataFrame Creation (`filtered_df`)**:
   - Returns a filtered DataFrame with only the specified columns.

4. **State ID Addition (`add_state_id`)**:
   - Adds masked state identifiers by merging with a CSV mapping file.
   - Saves the combined dataset to Parquet format.

5. **State-Based File Separation (`separate_to_states`)**:
   - Separates the dataset into individual files based on unique state identifiers.
   - Saves each state's data as a separate Parquet file.

6. **Unique State Extraction (`unique_states`)**:
   - Extracts unique state identifiers from a CSV file for downstream tasks.

7. **State-Based Data Loading (`load_states`)**:
   - Dynamically loads data for specified states into a dictionary for easy access.

---

### **`preprocessing.py`**
This module provides a comprehensive toolkit for processing and analyzing time series data. The methods include cleaning, scaling, transforming, and extracting trends from time series data, with support for both **Pandas** and **Dask** DataFrames.

#### Features

1. **Subset Data (`subset_data`)**:
   - Extracts a subset of the dataset based on a specific column value.

2. **Time Series Extraction (`get_time_series`)**:
   - Converts a dataset into a time series by parsing and normalizing date and time information.

3. **Outlier Removal (`remove_outliers`)**:
   - Detects and removes outliers in a specified column using the Interquartile Range (IQR) method.
   - Replaces outliers with the column's mean.

4. **Data Scaling (`scale_data`)**:
   - Scales data in a specified column to the range [0, 1] using Min-Max scaling.

5. **Remove Zero Values (`drop_zeros`)**:
   - Removes rows where the specified column has a value of zero.

6. **Processed Time Series Creation (`get_processed_series`)**:
   - Combines multiple processing steps:
     - Extracts a time series.
     - Removes outliers.
     - Scales the data.
     - Drops rows with zero values.

7. **Train-Test Split (`time_series_split`)**:
   - Splits the time series into training and testing datasets, preserving temporal order.

8. **Weekly Data Conversion (`convert_to_weekly`)**:
   - Aggregates the time series data into weekly intervals.

9. **Monthly Data Conversion (`convert_to_monthly`)**:
   - Aggregates the time series data into monthly intervals.

10. **Trend Extraction (`get_trend`)**:
    - Calculates a smoothed trend for the time series using a rolling mean with a specified window size.

###forecasting.py
###plotting.py

## Conclusion


