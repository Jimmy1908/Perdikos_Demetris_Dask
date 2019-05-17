#Importing Libraries
import dask.dataframe as dd
import numpy as np
import pandas as pd
import dask.bag as db
import matplotlib.pyplot as plt
from dask.distributed import client
from sklearn import preprocessing
import dask.array as da
import dask.array.stats
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from scipy.stats import skew, boxcox_normmax
from scipy.special import boxcox1p
from dask_ml.preprocessing import Categorizer, DummyEncoder, RobustScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from dask_ml.preprocessing import OneHotEncoder
from dask_ml.model_selection import train_test_split
from dask_ml.linear_model import LinearRegression
from dask_ml.metrics import mean_absolute_error
from dask_ml.metrics import mean_squared_error
from sklearn.metrics import r2_score

#Data loading
day_data = dd.read_csv(
    "https://gist.githubusercontent.com/geraldwal/483038a2a588e29dbad6ec4e51bbb6bc/raw/152607f899f5746a1b6afa103f68052ffe3fef65/day%2520-%2520Python%2520Bike%2520Sharing",
    sep=","
)
hour_data = dd.read_csv(
    "https://gist.githubusercontent.com/geraldwal/b5a83f4c670abe0a662abce558e5d433/raw/bce4bbfc63355606e4503964e25798b5d2190b9b/hour%2520-%2520Python%2520Bike%2520Sharing",
    sep=",",
    parse_dates=[1]
)
location_data = dd.read_csv(
    "https://gist.githubusercontent.com/geraldwal/3aba0d1a34965e2e59e1b6bf37e0bd91/raw/0266a0f936f2e1e75495d2e351090e80b49c107a/LocationsCapitalBikeStations%2520-%2520Python%2520Group",
    sep=","
)


#Rename Columns
hour_data=hour_data.rename(
    columns={
        "weathersit": "weather",
        "mnth": "month",
        "hr": "hour",
        "hum": "humidity",
        "cnt": "count",
        "yr": "year",
    })


#Change Date Format
hour_data.dteday = dd.to_datetime(hour_data.dteday, format="%Y-%m-%d")
hour_data["day"] = hour_data["dteday"].dt.day
hour_data=hour_data.drop('dteday',axis=1)
display(hour_data.head())

#Exploratory Data Analysis
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))

ax1 = axs[0]
ax1.scatter(
    hour_data["temp"].compute(),
    hour_data["casual"].compute(),
    c=hour_data["temp"].compute(),
    cmap="RdYlBu_r",
)
ax1.set_xlabel("Temperature")
ax1.set_ylabel("Casually Rented Bikes")
ax1.set_title("Casual Bike Users vs Temperature")

ax2 = axs[1]
ax2.scatter(
    hour_data["atemp"].compute(),
    hour_data["casual"].compute(),
    c=hour_data["atemp"].compute(),
    cmap="RdYlBu_r",
)
ax2.set_xlabel("Felt temperature")
ax2.set_ylabel("Casually Rented Bikes")
ax2.set_title("Casual Bike Users vs Felt Temperature")

plt.show()


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))

ax1 = axs[0]
ax1.scatter(
    hour_data["temp"].compute(),
    hour_data["registered"].compute(),
    c=hour_data["temp"].compute(),
    cmap="RdYlBu_r",
)
ax1.set_xlabel("Temperature")
ax1.set_ylabel("Registered Rented Bikes")
ax1.set_title("Registered Bike Users vs Temperature")

ax2 = axs[1]
ax2.scatter(
    hour_data["atemp"].compute(),
    hour_data["registered"].compute(),
    c=hour_data["atemp"].compute(),
    cmap="RdYlBu_r",
)
ax2.set_xlabel("Felt temperature")
ax2.set_ylabel("Registered Rented Bikes")
ax2.set_title("Registered Bike Users vs Felt Temperature")

plt.show()

#Normalize the data
ss = StandardScaler()
norm = ss.fit_transform(
    hour_data.drop(
        [
            "season",
            "year",
            "month",
            "hour",
            "holiday",
            "weekday",
            "workingday",
            "weather",
            "day",
            "casual",
            "registered",
            "count",
        ],
        axis=1,
    )
)
cat = hour_data.loc[
    :,
    [
        "season",
        "year",
        "month",
        "hour",
        "holiday",
        "weekday",
        "workingday",
        "weather",
        "day",
        "casual",
        "registered",
        "count",
    ],
]
hour_data = dd.concat([cat, norm], axis=1)


#One-Hot Encode Categorical Variables
hour_data_cl = hour_data.copy()
pipe = make_pipeline(
    Categorizer(), DummyEncoder())

hour_data_cl=pipe.fit_transform(hour_data_cl)

#Rename columns
hour_data_cl.rename(
    columns={
        "holiday_0": "no_holiday",
        "holiday_1": "yes_holiday",
        "month_1": "jan",
        "month_2": "feb",
        "month_3": "mar",
        "month_4": "apr",
        "month_5": "may",
        "month_6": "jun",
        "month_7": "jul",
        "month_8": "aug",
        "month_9": "sep",
        "month_10": "oct",
        "month_11": "nov",
        "month_12": "dec",
        "weekday_0": "sun",
        "weekday_1": "mon",
        "weekday_2": "tue",
        "weekday_3": "wed",
        "weekday_4": "thu",
        "weekday_5": "fri",
        "weekday_6": "sat",
        "season_1": "winter",
        "season_2": "spring",
        "season_3": "summer",
        "season_4": "autumn",
    }
)


#Function to split time series data
def split_data(dataset, Target):
    X = dataset.loc[:, dataset.columns != Target]
    y = dataset.loc[:, Target]
    train_size = int(len(dataset) * 0.875)
    X_train, X_test, y_train, y_test = (
        X.loc[0:train_size],
        X.loc[train_size : len(dataset)],
        y.loc[0:train_size],
        y.loc[train_size : len(dataset)],
    )
    return X_train, X_test, y_train, y_test

#Split the Dataset with 3 different targets (Registered, Casual, Count)
x_train_reg, x_test_reg, y_train_reg, y_test_reg = split_data(hour_data_cl, "registered")
x_train_casual, x_test_casual, y_train_casual, y_test_casual = split_data(hour_data_cl, "casual")
x_train_count, x_test_count, y_train_count, y_test_count = split_data(hour_data_cl, "count")

#drop target
x_train_reg=x_train_reg.drop(['casual','count'],axis=1)
x_test_reg=x_test_reg.drop(['casual','count'],axis=1)
x_train_casual=x_train_casual.drop(['registered','count'],axis=1)
x_test_casual=x_test_casual.drop(['registered','count'],axis=1)
x_train_count=x_train_count.drop(['registered','casual'],axis=1)
x_test_count=x_test_count.drop(['registered','casual'],axis=1)


#tranform into dask array to be used in ML algorithms
x_train_reg=x_train_reg.values
x_test_reg=x_test_reg.values
y_train_reg=y_train_reg.values
y_test_reg=y_test_reg.values

x_train_casual=x_train_casual.values
x_test_casual=x_test_casual.values
y_train_casual=y_train_casual.values
y_test_casual=y_test_casual.values

x_train_count=x_train_count.values
x_test_count=x_test_count.values
y_train_count=y_train_count.values
y_test_count=y_test_count.values

#Define a function to fit Linear Regression
def score_lin(X_train, X_test, y_train, y_test):
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    y_pred = lm.predict(X_test)
    print("Intercept:", lm.intercept_)
    print("Coefficients:", lm.coef_)
    print("Mean squared error (MSE): {:.2f}".format(mean_squared_error(y_test, y_pred)))
    print("Mean absolute error (MSE): {:.2f}".format(mean_absolute_error(y_test, y_pred)))
    print("Variance score (R2): {:.2f}".format(r2_score(y_test.compute(), y_pred.compute())))

    return y_pred

#Modelling
#Predicting the registered users
baseline_reg=score_lin(x_train_reg,x_test_reg,y_train_reg,y_test_reg)
baseline_reg
#Predicting the casual users
baseline_cas=score_lin(x_train_casual,x_test_casual,y_train_casual,y_test_casual)
baseline_cas
#Predicting total count
baseline_cnt=score_lin(x_train_count,x_test_count,y_train_count,y_test_count)
baseline_cnt
#The approach here was to try 2 ways of modelling:
#1)Predict Registered and Casual customers individually and independently and then sum the results
#2)Predict the final count directly

#Summing Registered and Casual predictions
pred_summed = baseline_reg + baseline_cas

#define scoring function to compare 2 approaches
def mae_score(x,y):
    s = mean_absolute_error(x,y)
    return s

#score 
mae_score( y_test_count ,pred_summed)

r2_score( y_test_count.compute() ,pred_summed.compute())
