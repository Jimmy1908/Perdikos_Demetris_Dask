#Importing Libraries
import dask.dataframe as dd
import numpy as np
import pandas as pd
import dask.bag as db
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
from dask_ml.metrics import r2_score

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
