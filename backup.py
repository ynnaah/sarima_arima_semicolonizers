from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_log_error
import itertools
import warnings

#read data
df = pd.read_csv('data.csv')
df.head()

#plt.figure(figsize=[12, 10]); # Set dimensions for figure
#df.plot(x='Time', y='People', figsize = (14, 6), legend = True, color='g')
#plt.title('Accomodation Demand, Library, Hourly')
#plt.ylabel('Number of People')
#plt.xlabel('Operational Hours')
#plt.grid(True)
#plt.show()

#differencing
df['People First Difference'] = df['People'] - df['People'].shift(1)
df.dropna(subset = ["People First Difference"], inplace=True)
df.head()

# Augmented Dickey-Fuller test (ADF Test)
ad_fuller_result = adfuller(df['People'])
print(f'ADF Statistic: {ad_fuller_result[0]}')
print(f'p-value: {ad_fuller_result[1]}')


best_model = SARIMAX(df['People'], order=(2, 1, 2), seasonal_order=(2, 1, 2, 12)).fit(dis=-1)
print(best_model.summary())

#Forecasting 24 epoch ahead
forecast_values = best_model.get_forecast(steps = 24)

#Confidence intervals of the forecasted values
forecast_ci = forecast_values.conf_int()

#Plot the data
ax = df.plot(x='Time', y='People', figsize = (14, 6), legend = True, color='purple')

#Plot the forecasted values 
forecast_values.predicted_mean.plot(ax=ax, label='Forecast', figsize = (14, 6), grid=True)

#Plot the confidence intervals
ax.fill_between(forecast_ci.index,
                forecast_ci.iloc[: , 0],
                forecast_ci.iloc[: , 1], color='yellow', alpha = .5)
plt.title('Accomodation Demand, Library, Hourly')
plt.ylabel('Number of People')
plt.legend(loc='upper left', prop={'size': 12})
ax.axes.get_xaxis().set_visible(True)
#annotation
ax.text(40, 500, 'Forecasted Values Until ', fontsize=12,  color='red')
plt.show()

