(This is still a work in progress)

# Forecasting Electricity Demand in Argentina

This project is to analyze the electricity demand in Argentina and to make a Time Series Forecast model that could predict 12 months forward.

- `eda.ipynb` is the analysis of the Time series.
- `SARIMAX.ipynb` is where we start to use the SARIMAX model approach to discover which combination of parameters performs best. We attempt to find them manually and also with pmdarima (http://alkaline-ml.com/pmdarima/).
- `machine_learning.ipynb` is where we explore the combination of different machine learning models.

By now, the model which performs best is SARIMAX(0,0,1)(1,1,0)6, accounting Bussiness Days as the Exogenous variable. With this model, we have an r2 score of 86.67% and a MAPE of 19.13%

!["Complete data and forecast"](imgs/img1.png?raw=true "Complete data and forecast")
!["Only Forecast w/ Prediction Intervals"](imgs/img2.png?raw=true "Only Forecast w/ Prediction Intervals")
