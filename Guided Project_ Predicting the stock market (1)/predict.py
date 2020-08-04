import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
stocks = pd.read_csv('sphist.csv')
#print(stocks.head())
#print(stocks.info())
stocks['Date']=pd.to_datetime(stocks['Date'])
#print(stocks.info())
date_bool = stocks['Date'] > datetime(year=1951, month=1, day=2)
stocks=stocks.sort_values(by=['Date'])
#print(stocks.head(2))
#indicator = [5,30,365]
# for index, row in stocks.iterrows():
#     s_5=0
#     s_30=0
#     s_365=0
#     #print(index)
#     for i in range(5):
#         s_5=s_5 + stocks.loc[index-i,'Close']
#     stocks.loc[index-5,'close_5_it'] = s_5/5
#     for i in range(30):
#         s_30=s_30+stocks.loc[index-i,'Close']
#     stocks.loc[index-30,'close_30_it'] = s_30/30
#     for i in range(365):
#         s_365=s_365+stocks.loc[index-i,'Close']
#     stocks.loc[index-365,'close_365_it'] = s_365/365
#df['Volume'].rolling(window=5).mean().shift(1)
stocks['Close_5_ro'] = stocks['Close'].rolling(window=5).mean().shift(1)
stocks['Close_30_ro'] = stocks['Close'].rolling(window=30).mean().shift(1)
stocks['Close_365_ro'] = stocks['Close'].rolling(window=365).mean().shift(1)
stocks=stocks[date_bool]
stocks = stocks.dropna(axis=0)
test=stocks[stocks['Date'] >= datetime(year=2013, month=1, day=1)]
train=stocks[stocks['Date']<datetime(year=2013, month=1, day=1)]
#print(stocks.head())
#print(test.head())
#print(train.head())
model=LinearRegression()

features1 = ['Open', 'High', 'Low', 'Volume', 'Adj Close', 'Close_5_ro']
features2 = ['Open', 'High', 'Low', 'Volume', 'Adj Close', 'Close_30_ro']
features3 = ['Open', 'High', 'Low', 'Volume', 'Adj Close', 'Close_365_ro']

target = ['Close']

model.fit(train[features1], train[target])
predictions1 = model.predict(test[features1])

model.fit(train[features2], train[target])
predictions2 = model.predict(test[features2])

model.fit(train[features3], train[target])
predictions3 = model.predict(test[features3])

mse_5 = mean_squared_error(test[target], predictions1)
mse_30 = mean_squared_error(test[target], predictions2)
mse_365 = mean_squared_error(test[target], predictions3)

print(mse_5,mse_30,mse_365)