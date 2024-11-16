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

![]() <img src="../call_volumes0JTMM.png" width=95%>

Plotting the time series data for the call volume, we see that the data has a lot of instability. In particular, we see increases in call volumes between years, which is likely due to more EMS centers reporting data over time. To account for this, we normalize the call volume for each year to obtain a more stationary data set, which is necessary in order to fit SARIMA models.

![]() <img src="../scaledcallvol.png" width=95%>

## Modeling
### SARIMA Models

Based on the EDA, there seems to be a seasonal component to the data, which can be captured by a SARIMA model. Since each county appeared to have different behaviour, we decided to train models on single counties. Here, we describe the process for producing a model for a single county.

We perform analysis on the autocorrelation and partial autocorellation functions in order to deduce reasonable parameters for our SARIMA(p,d,q)(P,D,Q)[m] model. 

<p float="left">
  <img src="../autocorr0JTMM.png" width="48%" />
  <img src="../partialautocorr0JTMM.png" width="48%" /> 
</p>

- The ACF and PACF both have significant lags at multiples of 7, which suggests that we have a seasonality of m=7.

Using the Akaike information Criterion (AIC), which estimates prediction error and allows us to weigh the relative quality of certain models, we perform our model selection by doing a stepwise search among parameters.  We select three models with low AIC values and compare these models against a baseline naive model using cross validation.

![]() <img src="../modelcomparison.png" width=95%>

Here are the mean squared errors for these four models:
| Model   | MSE |
| -------- | ------- |
| Naive model                | 0.05129    |
| SARIMA(0,1,1)(3,0,1)[7]    | 0.03308    |
| SARIMA(1,1,1)(2,0,1)[7]    | 0.03244    |
| SARIMA(0,1,2)(2,0,1)[7]    | 0.03240    |

The naive model had the largest MSE, and all three selected SARIMA models cut the error by over a third. The SARIMA(1,2,2)(2,0,1)[7] performed the best.

### Neural Network

## Conclusion


