# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 1/11/2025


### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

# Load IMDb dataset
data = pd.read_csv("IMDB Top 250 Movies (1).csv")

# Preprocess
data = data[['year', 'rating']].copy()
data['year'] = pd.to_numeric(data['year'], errors='coerce')
data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
data.dropna(inplace=True)
data = data.groupby('year')['rating'].mean().reset_index()
data.columns = ['Year', 'Rating']
data['Year'] = pd.to_datetime(data['Year'], format='%Y')
data.set_index('Year', inplace=True)

print("Shape of the dataset:", data.shape)
print("First 10 rows of the dataset:")
print(data.head(10))

# Plot original
plt.figure(figsize=(12, 6))
plt.plot(data['Rating'], label='Original IMDb Rating Data')
plt.title('Original IMDb Ratings Data')
plt.xlabel('Year')
plt.ylabel('Average IMDb Rating')
plt.legend()
plt.grid()
plt.show()

# Moving Averages
rolling_mean_5 = data['Rating'].rolling(window=5).mean()
rolling_mean_10 = data['Rating'].rolling(window=10).mean()

plt.figure(figsize=(12, 6))
plt.plot(data['Rating'], label='Original Data', color='blue')
plt.plot(rolling_mean_5, label='Moving Average (window=5)')
plt.plot(rolling_mean_10, label='Moving Average (window=10)')
plt.title('Moving Average of IMDb Ratings')
plt.xlabel('Year')
plt.ylabel('Average IMDb Rating')
plt.legend()
plt.grid()
plt.show()

# Add +1 to handle multiplicative seasonality
data['Rating'] = data['Rating'] + 1

# Train-test split
x = int(len(data) * 0.8)
train_data = data['Rating'][:x]
test_data = data['Rating'][x:]

# Exponential Smoothing (additive trend, multiplicative seasonality)
seasonal_periods = max(2, min(4, len(train_data)//2))
model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=seasonal_periods).fit()
test_predictions_add = model_add.forecast(steps=len(test_data))

# Visual evaluation
ax = train_data.plot(figsize=(12, 6))
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add", "test_data"])
ax.set_title('Visual Evaluation - Exponential Smoothing on IMDb Ratings')
plt.show()

# Evaluate RMSE
rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print("RMSE on test data:", rmse)

# Full model prediction for future
model_full = ExponentialSmoothing(data['Rating'], trend='add', seasonal='mul', seasonal_periods=seasonal_periods).fit()
predictions = model_full.forecast(steps=int(len(data)/4))

ax = data['Rating'].plot(figsize=(12, 6))
predictions.plot(ax=ax)
ax.legend(["data_yearly", "predictions"])
ax.set_xlabel('Year')
ax.set_ylabel('IMDb Rating (+1 shifted)')
ax.set_title('Exponential Smoothing Forecast for IMDb Ratings')
plt.show()

```

### OUTPUT:
<img width="1366" height="825" alt="image" src="https://github.com/user-attachments/assets/1c71fd88-46e3-4e57-a8f1-c189371c0d5e" />


Moving Average
<img width="1442" height="655" alt="Screenshot 2025-11-01 105515" src="https://github.com/user-attachments/assets/feffd794-c51b-4595-a8cf-e8635a6f89a4" />


Exponential Smoothing
<img width="1457" height="638" alt="Screenshot 2025-11-01 105543" src="https://github.com/user-attachments/assets/3ead20b2-745f-4d01-b275-61f583b998b9" />

<img width="1217" height="594" alt="Screenshot 2025-11-01 105528" src="https://github.com/user-attachments/assets/8e6befb8-97ba-4163-8b15-90df91cac64b" />



### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
