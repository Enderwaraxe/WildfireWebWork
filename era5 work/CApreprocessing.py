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

us = united_states.UnitedStates()

DM = xr.open_dataset('Datasets\era5\GDM.nc')
soilmoisture = xr.open_dataset("Datasets\era5\SoilMoistureNLDAS(2000-2020)(SoilM_layer1).nc")
wind = xr.open_dataset("Datasets\era5\wind2000-2020.nc")#3600, 1801, global
lai = xr.open_dataset("Datasets\era5\lai2000-2020.nc")#3600, 1801, global
pressure = xr.open_dataset("Datasets\era5\pressure2000-2020.nc")#3600, 1801, global
rain = xr.open_dataset("Datasets\era5/rain2000-2020.nc")#3600, 1801, global
radiation = xr.open_dataset("Datasets\era5/radiation2000-2020.nc")#3600, 1801, global
temp = xr.open_dataset("Datasets\era5/temperature2000-2020.nc")#3600, 1801, global
biomass = xr.open_dataset("Datasets\era5/biomass.nc")#720, 360, global
lightning = xr.open_dataset("Datasets\era5\wglc_timeseries_30m_monthly_2010_2023.nc")#720, 360, global
pop = xr.open_dataset("Datasets\era5\population_1850-2100.nc")#720, 360, global

# soilmoisture = soilmoisture.assign_coords(longitude =(((soilmoisture.longitude + 180) % 360) - 180)).sortby('longitude')
wind = wind.assign_coords(longitude =(((wind.longitude + 180) % 360) - 180)).sortby('longitude')
lai = lai.assign_coords(longitude =(((lai.longitude + 180) % 360) - 180)).sortby('longitude')
pressure = pressure.assign_coords(longitude =(((pressure.longitude + 180) % 360) - 180)).sortby('longitude')
rain = rain.assign_coords(longitude =(((rain.longitude + 180) % 360) - 180)).sortby('longitude')
radiation = radiation.assign_coords(longitude =(((radiation.longitude + 180) % 360) - 180)).sortby('longitude')
temp = temp.assign_coords(longitude =(((temp.longitude + 180) % 360) - 180)).sortby('longitude')
lightning = lightning.reindex(lat=list(reversed(lightning.lat)))
soilmoisture = soilmoisture.reindex(lat=list(reversed(soilmoisture.lat)))
pop = pop.reindex(lat=list(reversed(pop.lat)))

# print(soilmoisture)
# print(wind)
# print(lai)
# print(pressure)
# print(rain)
# print(radiation)
# print(temp)
# print(biomass)
# print(lightning)
# print(lightning.density, lightning.power_median, lightning.power_mean, lightning.power_SD)
# print(pop)

# #Making the Global Emission dataset into a CA only Dataset
datasets = [DM, soilmoisture, wind, wind, lai, lai, pressure, rain, radiation, temp, biomass, lightning, lightning, lightning, pop]
varnames = ["DM","SoilM_layer1", "u10", "v10", "lai_hv", 'lai_lv', "sp", "tp", "ssrd", "t2m", "biomass", "density", "power_mean", "power_SD", "hdm"]
# datasets = [DM, soilmoisture, wind, wind, lai, lai, pressure, rain, radiation, temp, biomass, pop]
# varnames = ["DM", "swvl1", "u10", "v10", "lai_hv", 'lai_lv', "sp", "tp", "ssrd", "t2m", "biomass","hdm"]
CAdatasets = []
baselineId = 0
baseline = None
biomassid = 10
popid = 14
lightningids = [11,12,13]
aggregate = False

for index in range(0,len(datasets)):
    file = datasets[index]
    #reducing the file to only coordinates in the bounding box of CA, bounding box nums found online
    try:
        fileCAbound = file[varnames[index]][:,:,:].where((file['longitude']>=-124.410607) & (file['longitude']<=-114.134458) & (file['latitude']<=42.009659) & (file['latitude']>=32.534231), drop=True)
    except Exception:
        fileCAbound = file[varnames[index]][:,:,:].where((file['lon']>=-124.410607) & (file['lon']<=-114.134458) & (file['lat']<=42.009659) & (file['lat']>=32.534231), drop=True)
    CAdatasets.append(fileCAbound)
    if(index == baselineId):
        baseline = copy.deepcopy(fileCAbound)

for index in range(0,len(CAdatasets)):
    file = CAdatasets[index]
    CAdatasets[index] = file[:,:,:].where((file['time.year'] >= baseline['time.year'][0].values) & (file['time.year']<= baseline['time.year'][-1].values), drop=True)

try:
    baseline = baseline.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude')
except Exception:
    baseline = baseline.rio.set_spatial_dims(x_dim='lon',y_dim='lat')
baseline = baseline.rio.write_crs('epsg:4326')

scalefactor = 1
if(aggregate):
    scalefactor = 1/2
new_width = int(baseline.rio.width * scalefactor)
new_height = int(baseline.rio.height * scalefactor)
baseline = baseline.rio.reproject(
    baseline.rio.crs,
    shape=(new_height, new_width),
    resampling=Resampling.average)
print(baseline)

for index in range(0, len(datasets)):
    try:
        temp = CAdatasets[index].rio.set_spatial_dims(x_dim='longitude',y_dim='latitude')
    except Exception:
        temp = CAdatasets[index].rio.set_spatial_dims(x_dim='lon',y_dim='lat')

    new = temp.rio.write_crs('epsg:4326')
    reprojected = new.rio.reproject_match(baseline, Resampling = Resampling.average)
    CAdatasets[index] = reprojected
    # plt.matshow(reprojected[0])
    # plt.title(varnames[index])
    # plt.show()

for id in lightningids:
    extendedlightning = []
    for year in range(baseline['time.year'][0].values, CAdatasets[id]['time.year'][0].values):
        for month in range(0,12):
            extendedlightning.append(CAdatasets[id][month])
    extendedlightning.extend(CAdatasets[id])
    extendedlightning = np.array(extendedlightning)
    CAdatasets[id] = xr.DataArray(extendedlightning, dims=['time', 'y', 'x'],  coords={'x':CAdatasets[id]['x'].values, 'y':CAdatasets[id]['y'].values, })

extendedbiomass = CAdatasets[biomassid]
for year in range(CAdatasets[biomassid]['time.year'][-1].values, baseline['time.year'][-1].values):
    extendedbiomass=np.concatenate([extendedbiomass,[CAdatasets[biomassid][-1]] ])
CAdatasets[biomassid] = xr.DataArray(extendedbiomass, dims=['time', 'y', 'x'],  coords={'x':CAdatasets[biomassid]['x'].values, 'y':CAdatasets[biomassid]['y'].values, })

biomassmonthly = []
for year in range(0, len(CAdatasets[biomassid].time)):
    for month in range(0,12):
        biomassmonthly.append(CAdatasets[biomassid][year])
biomassmonthly = np.array(biomassmonthly)
CAdatasets[biomassid] = xr.DataArray(biomassmonthly, dims=['time', 'y', 'x'],  coords={'x':CAdatasets[biomassid]['x'].values, 'y':CAdatasets[biomassid]['y'].values, })

popmonthly = []
for year in range(0, len(CAdatasets[popid].time)):
    for month in range(0,12):
        popmonthly.append(CAdatasets[popid][year])
popmonthly = np.array(popmonthly)
CAdatasets[popid] = xr.DataArray(popmonthly, dims=['time', 'y', 'x'],  coords={'x':CAdatasets[popid]['x'].values, 'y':CAdatasets[popid]['y'].values, })


templat = []
templon = []
for latind in range(0, len(baseline['y'])):
   temptemplat = []
   temptemplon = []
   for lonind in range(0, len(baseline['x'])):
      temptemplon.append(baseline['x'][lonind].values)
      temptemplat.append(baseline['y'][latind].values)
   templat.append(temptemplat)
   templon.append(temptemplon)   
      
monthsin = []
monthcos = []
lat = []
lon = []     

for time in range(0, len(baseline['time.month'])):
   tempsin = np.empty((len(baseline['y']), len(baseline['x'])))
   tempsin.fill(np.sin((2*np.pi*int(baseline['time.month'][time].values))/12))
   tempcos = np.empty((len(baseline['y']), len(baseline['x'])))
   tempcos.fill(np.cos((2*np.pi*int(baseline['time.month'][time].values))/12))
   monthsin.append(tempsin)
   monthcos.append(tempcos)
   lat.append(templat)
   lon.append(templon)

monthsin, monthcos, lat, lon = np.array(monthsin), np.array(monthcos), np.array(lat), np.array(lon)

arrs = []
arrs.append(np.reshape(monthsin, (-1, len(baseline['x']),1)))
arrs.append(np.reshape(monthcos, (-1, len(baseline['x']),1)))
arrs.append(np.reshape(lat, (-1, len(baseline['x']),1)))
arrs.append(np.reshape(lon, (-1, len(baseline['x']),1)))

for i in range(1,len(CAdatasets)):
    print(varnames[i])
    print(np.array(CAdatasets[i]).shape)
    arrs.append(np.reshape(np.array(CAdatasets[i]), (-1, len(CAdatasets[i]['x']),1)))
arrs = np.array(arrs)


final = np.concatenate(arrs, axis=2)
final = np.reshape(final, (-1, len(baseline['y']), len(baseline['x']), len(arrs)))
print(final.shape)

Femissions = CAdatasets[0]


timelen,latlen,lonlen,varlen = final.shape
for latid in range(0,latlen):
    for lonid in range(0, lonlen):
        result = us.from_coords(baseline['y'][latid].values,baseline['x'][lonid].values)
        print(result)
        if (len(result)==0 or result[0].abbr != "CA"):
            final[:, latid, lonid] = [np.nan]*varlen
            Femissions[:, latid,lonid] = np.nan
# for var in range(0, 15):
#     plt.matshow(final[0, :, :, var])
#     plt.show()
# for time in range(0, 10):
#     plt.matshow(final[time, :, :, 0])
#     plt.show()
# print(final)

Femissions = Femissions.rename({'y':"lat", "x":"lon"})
variables = []
variables.append('month sin')
variables.append('month cos')
variables.append('lat')
variables.append('lon')
for i in range(1, len(datasets)):
  variables.append(varnames[i])
print(final.shape, Femissions.values.shape, baseline['y'].shape, baseline['x'].shape)
CAdata = xr.Dataset(
    data_vars=dict(data=(['time','lat', 'lon', 'variables'], final), DM=(['time', 'lat','lon'],Femissions.values)),
coords=dict(time=(baseline.time), lat=(baseline['y'].values), lon=(baseline['x'].values),variables=(variables))
)
if(aggregate):
    CAdata.to_netcdf(path='./CAdata_gridded(era5)(2x2Aggregated).nc')
else:
    CAdata.to_netcdf(path='./CAdata_gridded(era5).nc')