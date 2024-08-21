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
GDM = []
time = []
for year in range(2002, 2021):
  temp = xr.open_dataset(f"Datasets/era5/GFED5/GFED5_Beta_monthly_{year}.nc")
  time.extend(temp.time.values)
  GDM.extend(temp.DM.values)

print(np.array(GDM).shape)
GDMdata = xr.Dataset(
    data_vars = dict(DM = (['time', 'lat', 'lon'],GDM )),
    coords =  dict(time = (time), lat = (temp.lat.values), lon = (temp.lon.values)),
    attrs= dict(subtitle = "GFED5 monthly dry matter emission data from 2002-2020")
)
print(GDMdata)
GDMdata.to_netcdf(path='GDM.nc')

radiation = xr.open_dataset("Datasets/era5/NonFinal/radiation-004-003.nc")
rad2020 = xr.open_dataset("Datasets/era5/NonFinal/radiation-2020.nc")

rain = xr.open_dataset("Datasets/era5/NonFinal/rain-002-004.nc")
rain2020 = xr.open_dataset("Datasets/era5/NonFinal/rain-2020.nc")

temp = xr.open_dataset("Datasets/era5/NonFinal/temperature-003-005.nc")
temp2020 = xr.open_dataset("Datasets/era5/NonFinal/temperature-2020.nc")

press = xr.open_dataset("Datasets/era5/NonFinal/pressure-005-002.nc")
press2020 = xr.open_dataset("Datasets/era5/NonFinal/pressure-2020.nc")

lai = xr.open_dataset("Datasets/era5/NonFinal/lai-006-007.nc")
lai2020 = xr.open_dataset("Datasets/era5/NonFinal/lai-2020.nc")

wind = xr.open_dataset("Datasets/era5/NonFinal/wind-001-001.nc")
wind2020 = xr.open_dataset("Datasets/era5/NonFinal/wind-2020.nc")

radf = xr.concat([radiation, rad2020], dim='time')
rainf = xr.concat([rain, rain2020], dim='time')
tempf = xr.concat([temp, temp2020], dim='time')
pressf = xr.concat([press, press2020], dim='time')
laif = xr.concat([lai, lai2020], dim='time')
windf = xr.concat([wind, wind2020], dim='time')


radf.to_netcdf(path='./radiation2000-2020.nc')
rainf.to_netcdf(path='./rain2000-2020.nc')
tempf.to_netcdf(path='./temperature2000-2020.nc')
pressf.to_netcdf(path='./pressure2000-2020.nc')
laif.to_netcdf(path='./lai2000-2020.nc')
windf.to_netcdf(path='./wind2000-2020.nc')

moisture = xr.open_dataset("Datasets/era5/Soilmoisture/NLDAS_VIC0125_M.A202012.020.nc")
soilmoisture = []
time = []
lat = moisture.lat.values
lon = moisture.lon.values
for year in range(2000, 2021):
    for month in range(1, 13):
        if(month<10):
            dataset = xr.open_dataset("Datasets/era5/Soilmoisture/NLDAS_VIC0125_M.A" + str(year) + "0" + str(month) +".020.nc")
        else: 
            dataset = xr.open_dataset("Datasets/era5/Soilmoisture/NLDAS_VIC0125_M.A" + str(year) + str(month) +".020.nc")
        soilmoisture.append(dataset["SoilM_total"][0].values)
        time.append(dataset.time[0].values)
time = np.array(time)
soilmoisture = np.array(soilmoisture)
print(time.shape)
print(soilmoisture.shape)
moisturedata = xr.Dataset(
    data_vars=dict(SoilM_total=(['time','lat', 'lon'], soilmoisture)),
    coords=dict(time=(time), lat=(lat), lon=(lon))
)
print(moisturedata)
moisturedata.to_netcdf(path='./SoilMoistureNLDAS(2000-2020).nc')

