import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

#Read the data and change the date column to datetime objects
df = pd.read_csv('sphist.csv')
df['Date'] = pd.to_datatime(df['Date'])
df = df.sort_values('Date', ascending = True)
df.reset_index(inplace = True) 

#Add indicators that could be helpful for machine learning

#The average price from the past 5 days.
df['avg_5'] = df['Close'].rolling(window=5).mean().shift(1)
#The average price from the past 30 days.
df['avg_30'] = df['Close'].rolling(window=30).mean().shift(1)
#The average price from the past 365 days.
df['avg_365'] = df['Close'].rolling(window=365).mean().shift(1)

#The std deviation of the price over the past 5 days.
df['std_5'] = df['Close'].rolling(window=5).std().shift(1)
#The std deviation of the price over the past 365 days
df['std_365'] = df['Close'].rolling(window=365).std().shift(1)

#The ratio between the average price for the past 5 days, and the average price for the past 365 days
df["avg_5/avg_365"] = df["avg_5"]/df["avg_365"]
#The ratio between the standard deviation for the past 5 days, and the standard deviation for the past 365 days.
df["std_5/std_365"] = df["std_5"]/df["std_365"]


#Some of the indicators use 365 days of historical data, and the dataset starts on 1950-01-03.
#Thus, any rows that fall before 1951-01-03 don't have enough historical data to compute all the indicators.
df = df[df['Date'] > datetime(year=1951, month=1, day=3)]
df.dropna(axis=0, inplace = True)

#Train Test Split
train = df[df['Date'] < datetime(year=2013, month=1, day=1)]
test = df[df['Date'] >= datetime(year=2013, month=1, day=1)]

#Leave out all of the original columns (Close, High, Low, Open, Volume, Adj Close, Date) when training your model. 
#These all contain knowledge of the future that you don't want to feed the model.
features = ["avg_5", "avg_30", "avg_365", "std_5", "std_365", "avg_5/avg_365", "std_5/std_365"]

model = LinearRegression()
model.fit(train[features], train['Close'])
prediction = model.predict(test[features])

#MAE will show you how "close" you were to the price in intuitive terms.
mae = mean_absolute_error(test['Close'],prediction)
print(mae)