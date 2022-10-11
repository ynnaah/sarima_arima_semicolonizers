from matplotlib import test
import pandas as pd
import numpy as np
import statsmodels as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

def testData(dataset):
    adf_test = adfuller(df_train)
    print(f'p-value: {adf_test[1]}')

df = pd.read_csv('data.csv')
df = np.log(df['People'])

msk = (df.index < len(df)-19)
df_train = df[msk].copy()
df_test = df[~msk].copy()
df_train_diff = df_train.diff().dropna()

model = ARIMA(df_train, order=(0,0,2))
model_fit = model.fit()
print(model_fit.summary())

