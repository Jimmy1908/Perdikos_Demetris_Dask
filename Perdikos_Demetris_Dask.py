#Importing Libraries
import dask.dataframe as dd
import numpy as np
import pandas as pd
import dask.bag as db
import matplotlib.pyplot as plt
from dask.distributed import Client
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
import matplotlib.pyplot as plt
import seaborn as sns
from dask_ml.xgboost import XGBRegressor
client = Client()

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

#Scatter Plots
#Casual Bike Users vs Temperature
#Casual Bike Users vs Felt Temperature
#Registered Bike Users vs Temperature
#Registered Bike Users vs Felt Temperature


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



#Trend

fig, ax = plt.subplots(1)
ax.plot(sorted(hour_data["casual"].compute()), c="b", marker="*", label="casual")
ax.plot(
    sorted(hour_data["registered"].compute()),
    c="#EE273A",
    marker="*",
    label="registered",
)
ax.legend(loc="upper left")
ax.set_ylabel("Sorted Rental Counts", fontsize=10)
fig.suptitle("Recorded Bike Rental Counts", fontsize=10)

plt.show()




#Scatter Plots
#Casual vs Windspeed
#Registered vs Windspeed
#Both plots mentioned above show an decreasing trend of bicycle use when the windspeed is increasing (both casual and registered bikes).

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))

ax1 = axs[0]
ax1.scatter(
    hour_data["windspeed"].compute(),
    hour_data["casual"].compute(),
    c=hour_data["windspeed"].compute(),
    cmap="autumn_r",
)
ax1.set_xlabel("windspeed")
ax1.set_ylabel("Casually Rented Bikes")
ax1.set_title("Casual v/s windspeed")

ax2 = axs[1]
ax2.scatter(
    hour_data["windspeed"].compute(),
    hour_data["registered"].compute(),
    c=hour_data["windspeed"].compute(),
    cmap="autumn_r",
)
ax2.set_xlabel("windspeed")
ax2.set_ylabel("Registered Rented Bikes")
ax2.set_title("Registered v/s windspeed")

plt.show()


#Scatter Plots
#Registered vs Humidity: Registered bikes use decrease as humidity increases.
#Casual vs Humidity: Casual bikes use is steady even when humidity increases.
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))

ax1 = axs[0]
ax1.scatter(
    hour_data["humidity"].compute(),
    hour_data["casual"].compute(),
    c=hour_data["humidity"].compute(),
    cmap="Blues",
)
ax1.set_xlabel("Humidity")
ax1.set_ylabel("Casually Rented Bikes")
ax1.set_title("Casual v/s humidity")

ax2 = axs[1]
ax2.scatter(
    hour_data["humidity"].compute(),
    hour_data["registered"].compute(),
    c=hour_data["humidity"].compute(),
    cmap="Blues",
)
ax2.set_xlabel("Humidity")
ax2.set_ylabel("Registered Rented Bikes")
ax2.set_title("Registered v/s humidity")

plt.show()


#Bar Chart
#Total Casual Bikes Rentals per Daily Hour: Increasing trend in bikes rental after 7am, decrease starts after 6pm.
#Total Registered Bikes Rentals per Daily Hour: Increasing trend in bikes rental after 6am, decrease starts after 8pm.

ax1 = (
    hour_data.compute()[["hour", "casual"]]
    .groupby(["hour"])
    .sum()
    .reset_index()
    .plot(
        kind="bar",
        figsize=(8, 6),
        legend=False,
        title="Total Casual Bike Rentals by Hour",
        color="#EE273A",
        fontsize=12,
    )
)
ax1.set_xlabel("Hour", fontsize=12)
ax1.set_ylabel("Casual", fontsize=12)

ax2 = (
    hour_data.compute()[["hour", "registered"]]
    .groupby(["hour"])
    .sum()
    .reset_index()
    .plot(
        kind="bar",
        figsize=(8, 6),
        legend=False,
        title="Total Registered Bike Rentals by Hour",
        color="#EE273A",
        fontsize=12,
    )
)
ax2.set_xlabel("Hour", fontsize=12)
ax2.set_ylabel("Registered", fontsize=12)

plt.show()




#Bar Charts
#Bike Rentals by Season: Highest number of casual bikes rented in fall, followed by summer and winter respectively.
#Bike Rentals by Weathersit: Highest number of casual bikes renter during clear weathersit.

f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

ax1 = (
    hour_data.compute()[["season", "casual"]]
    .groupby(["season"])
    .sum()
    .reset_index()
    .plot(
        kind="bar",
        legend=False,
        title="Bike Rentals by season",
        stacked=True,
        fontsize=12,
        ax=ax1,
        color="#EE273A",
    )
)
ax1.set_xlabel("season", fontsize=12)
ax1.set_ylabel("Casual Count", fontsize=12)
ax1.set_xticklabels(["spring", "summer", "fall", "winter"])


ax2 = (
    hour_data.compute()[["weather", "casual"]]
    .groupby(["weather"])
    .sum()
    .reset_index()
    .plot(
        kind="bar",
        legend=False,
        stacked=True,
        title="Bike Rentals by weathersit",
        fontsize=12,
        ax=ax2,
        color="#EE273A",
    )
)

ax2.set_xlabel("weather", fontsize=12)
ax2.set_ylabel("Casual Count", fontsize=12)
ax2.set_xticklabels(["1: Clear", "2: Mist", "3: Light Snow", "4: Heavy Rain"])

f.tight_layout
plt.show()











#Histograms
#Number of casual bikes rented per season: Both graphs compare bike rental per season during year 1 and year 2,
#showing fall is the season with higher number of casual bikes rentals. The season with lowest rentals is spring.
plt.figure(figsize=(11, 5))
sns.barplot(
    "year",
    "casual",
    hue="season",
    data=hour_data.compute(),
    palette="rainbow",
    ci=None,
    estimator=sum,
)
plt.legend(["Spring", "Summer", "Fall", "Winter"], loc="upper left")
plt.xlabel("Year")
plt.ylabel("Total number of bikes rented on Casual basis")
plt.title("Number of casual bikes rented per season")

plt.show()


#Number of registered bikes rented per season: Both graphs compare registered bike rental per season during year 1 and year 2,
#showing fall is the season with higher number of casual bikes rentals. The season with lowest rentals is spring.

plt.figure(figsize=(11, 5))
sns.barplot(
    "year",
    "registered",
    hue="season",
    data=hour_data.compute(),
    palette="rainbow",
    ci=None,
)
plt.legend(["Spring", "Summer", "Fall", "Winter"], loc="upper left")
plt.xlabel("Year")
plt.ylabel("Total number of bikes rented on registered basis")
plt.title("Number of registered bikes rented per season")


plt.show()

#Correlation Matrix: Highly correlated variables are:
#Windspeed and humidity Casual and humidity Registered and humidity Count and humidity

corrMatt = hour_data[
    ["temp", "atemp", "humidity", "windspeed", "casual", "registered", "count"]
].corr()

mask = np.array(corrMatt)
# Turning the lower-triangle of the array to false
mask[np.tril_indices_from(mask)] = False
fig, ax = plt.subplots()
sns.heatmap(corrMatt, mask=mask, vmax=0.8, square=True, annot=True, ax=ax)

plt.show()



#Correlation Matrix: Highly correlated variables are:
#Count and registered Year and Instant Count and casual

plt.figure(figsize=(20, 5))
mask = np.zeros_like(hour_data.corr().compute(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(hour_data.corr().compute(), cmap="RdBu_r", mask=mask, annot=True)

plt.show()


#Bar Charts
#Monthly Distribution (Registered)
#Monthly Distribution (Casual)
#Months with highest number of rentals (casual and registered) are May to October, yearly decrease starts in November.

fig, ax = plt.subplots()
sns.barplot(
    data=hour_data.compute()[["month", "registered"]],
    x="month",
    y="registered",
    ax=ax,
    estimator=sum,
)
ax.set(title="Monthly distribution")

fig, ax = plt.subplots()
sns.barplot(
    data=hour_data.compute()[["month", "casual"]],
    x="month",
    y="casual",
    ax=ax,
    estimator=sum,
)
ax.set(title="Monthly distribution")

plt.show()




#sns.Pointplots
#1) Use of the system by total users; hourly distribution, split to weekdays: Show peak hours at 8am and 5pm, office hours.
#2) Use of the system by casual users; hourly distribution, split to weekdays: Show most activity between 9am and 8pm.
#3) Use of the system by registered users; hourly distribution, split to weekdays: Show peak hours at 8am and 5pm, office hours.


plt.figure(figsize=(14, 4))
weekday_hr_gb = (
    hour_data.compute()
    .groupby(["weekday", "hour"])
    .mean()["count"]
    .reset_index(drop=False)
)
g = sns.pointplot(x="hour", y="count", hue="weekday", data=weekday_hr_gb)
g_legend = g.axes.get_legend()
g_labels = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
]
for t, l in zip(g_legend.texts, g_labels):
    t.set_text(l)
plt.title("Use of the system total; hourly distribution, split to weekdays")
plt.show()


plt.figure(figsize=(14, 4))
weekday_hr_gb = (
    hour_data.compute()
    .groupby(["weekday", "hour"])
    .mean()["casual"]
    .reset_index(drop=False)
)
g = sns.pointplot(x="hour", y="casual", hue="weekday", data=weekday_hr_gb)
g_legend = g.axes.get_legend()
g_labels = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
]
for t, l in zip(g_legend.texts, g_labels):
    t.set_text(l)
plt.title("Use of the system by casual users; hourly distribution, split to weekdays")
plt.show()
print(
    "weekdays- no real peaks, increase within the day. weekends- peaks are during the day"
)

plt.figure(figsize=(14, 4))
weekday_hr_gb = (
    hour_data.compute()
    .groupby(["weekday", "hour"])
    .mean()["registered"]
    .reset_index(drop=False)
)
g = sns.pointplot(x="hour", y="registered", hue="weekday", data=weekday_hr_gb)
g_legend = g.axes.get_legend()
g_labels = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
]
for t, l in zip(g_legend.texts, g_labels):
    t.set_text(l)
plt.title(
    "Use of the system by registered users; hourly distribution, split to weekdays"
)
plt.show()
print(
    "weekdays- peaks are mornings and evenings. weekends- peaks are during the day, nearly similar to count"
)


#Histograms
#1) Humidity Histogram
#2) Humidity Histogram with quantiles: Frequency increases as humidity increases
#3) Temperature Histogram
#4) Temperature Histogram with quantiles: Frequency increases in second and third quantiles, and decrease starts on fourth quantile
#5) Felt Temperature Histogram
#6) Felt Temperature Histogram with quantiles: Frequency increases in second and third quantiles, and decrease starts on fourth quantile
#7) Windspeed Histogram
#8) Windspeed Histogram with quantiles: Increase in first quantile, and subtle decrease until fourth quantile.

#1
quantile_list = [0, 0.25, 0.5, 0.75, 1.0]

fig, ax = plt.subplots()
hour_data.compute()["humidity"].hist(
    bins=30, color="#A9C5D3", edgecolor="black", grid=False
)
ax.set_title("Humidity Histogram", fontsize=12)
ax.set_xlabel("Humidity", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)

plt.show()

#2
quantiles = hour_data.compute()["humidity"].quantile(quantile_list)

fig, ax = plt.subplots()
hour_data.compute()["humidity"].hist(
    bins=30, color="#A9C5D3", edgecolor="black", grid=False
)
for quantile in quantiles:
    qvl = plt.axvline(quantile, color="#EE273A")
ax.legend([qvl], ["Quantiles"], fontsize=10)
ax.set_title("Humidity Histogram with Quantiles", fontsize=12)
ax.set_xlabel("Humidity", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)

plt.show()

#3
fig, ax = plt.subplots()
hour_data.compute()["temp"].hist(
    bins=30, color="#A9C5D3", edgecolor="black", grid=False
)
ax.set_title("Temperature Histogram", fontsize=12)
ax.set_xlabel("Temperature", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)

plt.show()

#4
quantiles = hour_data.compute()["temp"].quantile(quantile_list)

fig, ax = plt.subplots()
hour_data.compute()["temp"].hist(
    bins=30, color="#A9C5D3", edgecolor="black", grid=False
)
for quantile in quantiles:
    qvl = plt.axvline(quantile, color="#EE273A")
ax.legend([qvl], ["Quantiles"], fontsize=10)
ax.set_title("Temperature Histogram with Quantiles", fontsize=12)
ax.set_xlabel("Temperature", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)

plt.show()

#5
fig, ax = plt.subplots()
hour_data.compute()["atemp"].hist(
    bins=30, color="#A9C5D3", edgecolor="black", grid=False
)
ax.set_title("Felt Temperature Histogram", fontsize=12)
ax.set_xlabel("Felt Temp", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)

plt.show()

#6
quantiles = hour_data.compute()["atemp"].quantile(quantile_list)

fig, ax = plt.subplots()
hour_data.compute()["atemp"].hist(
    bins=30, color="#A9C5D3", edgecolor="black", grid=False
)
for quantile in quantiles:
    qvl = plt.axvline(quantile, color="#EE273A")
ax.legend([qvl], ["Quantiles"], fontsize=10)
ax.set_title("Felt Temperature Histogram with Quantiles", fontsize=12)
ax.set_xlabel("Felt Temperature", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)

plt.show()

#7
fig, ax = plt.subplots()
hour_data.compute()["windspeed"].hist(
    bins=30, color="#A9C5D3", edgecolor="black", grid=False
)
ax.set_title("Windspeed Histogram", fontsize=12)
ax.set_xlabel("Windspeed", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)

plt.show()

#8
quantiles = hour_data.compute()["windspeed"].quantile(quantile_list)

fig, ax = plt.subplots()
hour_data.compute()["windspeed"].hist(
    bins=30, color="#A9C5D3", edgecolor="black", grid=False
)
for quantile in quantiles:
    qvl = plt.axvline(quantile, color="#EE273A")
ax.legend([qvl], ["Quantiles"], fontsize=10)
ax.set_title("Windspeed Temperature Histogram with Quantiles", fontsize=12)
ax.set_xlabel("Windspeed", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)

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


#XGBoost apprach
#A new scoring function is created in order to try a new approach
def score_xgb(X_train, X_test, y_train, y_test):
    est = XGBRegressor()
    est.fit(X_train, y_train)
    y_pred = est.predict(X_test)
    print("Mean squared error (MSE): {:.2f}".format(mean_squared_error(y_test, y_pred)))
    print(
        "Mean absolute error (MSE): {:.2f}".format(mean_absolute_error(y_test, y_pred))
    )
    print(
        "Variance score (R2): {:.2f}".format(
            r2_score(y_test.compute(), y_pred.compute())
        )
    )
    return y_pred

#score the model
baseline_reg = score_xgb(x_train_reg, x_test_reg, y_train_reg, y_test_reg)
baseline_reg


