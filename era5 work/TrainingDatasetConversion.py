from sklearn.cluster import DBSCAN
from sklearn import svm
from sklearn.pipeline import Pipeline
import optuna
import netCDF4 as nc4
import cftime
import numpy as np
import xarray as xr
import datetime as dt
import nc_time_axis
import calendar
from datetime import timedelta
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import united_states
from sklearn import datasets
from skimage.measure import block_reduce
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import math
import sklearn as sk
import pandas as pd
# import smogn
import rioxarray as rio
import copy
import seaborn as sns
import tables
import h5py
import rioxarray as rio
import rasterio
from rasterio.enums import Resampling
aggregated = False
if (aggregated):
    Dataset = xr.open_dataset("Datasets\era5\CAdata_gridded(era5)(2x2Aggregated).nc")
else:
    Dataset = xr.open_dataset("Datasets\era5\CAdata_gridded(era5).nc")

time, lat, lon, vars = Dataset.data.shape
timeoffset = 1
df = pd.DataFrame(np.reshape(Dataset.data.values[timeoffset:], (-1, vars)), columns = Dataset['variables'].values)
df.insert(vars, "Current_DM", np.reshape(Dataset.DM.values[timeoffset:],-1))
df.insert(vars+1, "Next_DM", np.reshape(Dataset.DM.values[:-timeoffset],-1))
df = df.dropna()
print(df)
if(aggregated):
    df.to_csv("TrainingData(2x2Aggregated).csv", index=False)
else:
    df.to_csv("TrainingData.csv", index=False)