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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
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

#Creating a global dataset for BA and Emissions with dimensions (time, lat,lon), for emissions also including extra information for biomes
#loading files
# GFED_2000 =h5py.File('/Users/Joshua Ni/Downloads/GFED4.1s_2000.hdf5', 'r')
# GFED_2001 =h5py.File('/Users/Joshua Ni/Downloads/GFED4.1s_2001.hdf5', 'r')
# GFED_2002 =h5py.File('/Users/Joshua Ni/Downloads/GFED4.1s_2002.hdf5', 'r')
# GFED_2003 =h5py.File('/Users/Joshua Ni/Downloads/GFED4.1s_2003.hdf5', 'r')
# GFED_2004 =h5py.File('/Users/Joshua Ni/Downloads/GFED4.1s_2004.hdf5', 'r')
# GFED_2005 =h5py.File('/Users/Joshua Ni/Downloads/GFED4.1s_2005.hdf5', 'r')
# GFED_2006 =h5py.File('/Users/Joshua Ni/Downloads/GFED4.1s_2006.hdf5', 'r')
# GFED_2007 =h5py.File('/Users/Joshua Ni/Downloads/GFED4.1s_2007.hdf5', 'r')
# GFED_2008 =h5py.File('/Users/Joshua Ni/Downloads/GFED4.1s_2008.hdf5', 'r')
# GFED_2009 =h5py.File('/Users/Joshua Ni/Downloads/GFED4.1s_2009.hdf5', 'r')
# GFED_2010 =h5py.File('/Users/Joshua Ni/Downloads/GFED4.1s_2010.hdf5', 'r')
# GFED_2011 =h5py.File('/Users/Joshua Ni/Downloads/GFED4.1s_2011.hdf5', 'r')
# GFED_2012 =h5py.File('/Users/Joshua Ni/Downloads/GFED4.1s_2012.hdf5', 'r')
# GFED_2013 =h5py.File('/Users/Joshua Ni/Downloads/GFED4.1s_2013.hdf5', 'r')
# GFED_2014 =h5py.File('/Users/Joshua Ni/Downloads/GFED4.1s_2014.hdf5', 'r')
# GFED_2015 =h5py.File('/Users/Joshua Ni/Downloads/GFED4.1s_2015.hdf5', 'r')
# datasets = [GFED_2000, GFED_2001,GFED_2002,GFED_2003,GFED_2004,GFED_2005,GFED_2006,GFED_2007,GFED_2008,GFED_2009,GFED_2010,GFED_2011,GFED_2012,GFED_2013,GFED_2014,GFED_2015]
# monthindex = ['01','02','03','04','05','06','07','08','09','10','11','12']
# latg = GFED_2000['lat'][...,0]
# long = GFED_2000['lon'][0,...]
# GlobalDM = []
# timeg = []
# #creating list of all time points
# for i in range(2000, 2016):
#     for j in range(1, 13):
#         if(j<10):
#             timeg.append(np.datetime64(str(i) + '-0' +str(j) +'-01T00:00:00.000000000'))
#         else:
#             timeg.append(np.datetime64(str(i) + '-' +str(j) +'-01T00:00:00.000000000'))

# #creating 3D arrays of dim (time, lat, lon)
# for indd in range(0, len(datasets)):
#     for indm in range(0, len(monthindex)):
#         GlobalDM.append(datasets[indd]['emissions'][monthindex[indm]]["DM"][()])
# #creating global datasets(too shrink to CA later)

# globaldata = xr.Dataset(
#     data_vars=dict(Emissions=(['time','lat', 'lon'], GlobalDM)),
#     coords=dict(time=(timeg), lat=(latg), lon=(long))
# )
# globaldata.to_netcdf(path='./Global_Emissions(2000-2015).nc')

# moisture = xr.open_dataset("Datasets/era5/Soilmoisture/NLDAS_VIC0125_M.A202010.020.nc")
# soilmoisture = []
# time = []
# lat = moisture.lat.values
# lon = moisture.lon.values
# for year in range(2000, 2016):
#     for month in range(1, 13):
#         if(month<10):
#             dataset = xr.open_dataset("Datasets/era5/Soilmoisture/NLDAS_VIC0125_M.A" + str(year) + "0" + str(month) +".020.nc")
#         else: 
#             dataset = xr.open_dataset("Datasets/era5/Soilmoisture/NLDAS_VIC0125_M.A" + str(year) + str(month) +".020.nc")
#         soilmoisture.append(dataset["SoilM_0_100cm"][0].values)
#         time.append(dataset.time[0].values)
# time = np.array(time)
# soilmoisture = np.array(soilmoisture)
# print(time.shape)
# print(soilmoisture.shape)
# moisturedata = xr.Dataset(
#     data_vars=dict(SoilM_0_100cm=(['time','lat', 'lon'], soilmoisture)),
#     coords=dict(time=(time), lat=(lat), lon=(lon))
# )
# print(moisturedata)
# moisturedata.to_netcdf(path='./SoilMoistureNLDAS(2000-2015).nc')

GEm = xr.open_dataset("Datasets\era5\Global_Emissions(2000-2015).nc")
print(GEm)
EmUSbound = GEm['Emissions'][:,:,:].where((GEm['lon']>=-125.0011) & (GEm['lon']<=-66.945392) & (GEm['lat']<=49.382808) & (GEm['lat']>=24.9493), drop=True)
lat = EmUSbound.lat
lon = EmUSbound.lon
time = EmUSbound['time']
nonUScoords = np.load("NonUS.npy")
EmUSbound = EmUSbound.values
for (lat1,lon1) in nonUScoords:
    EmUSbound[:,lat1,lon1].fill(np.nan)
print(EmUSbound.shape)
EmUS = xr.Dataset(
    data_vars=dict(y = (['time', 'lat', 'lon'], EmUSbound),),
    coords=dict(lon=(lon), lat=(lat), time=(time)),
    attrs= GEm.attrs
)
plt.matshow(EmUS['y'][0])



# EmUS.to_netcdf(path='./EmUS.nc')






#loading files
soilmoisture = xr.open_dataset("Datasets/era5/SoilMoistureNLDAS(2000-2015).nc")
wind = xr.open_dataset("Datasets\era5\wind-001-001.nc")
lai = xr.open_dataset("Datasets\era5\lai-006-007.nc")
pressure = xr.open_dataset("Datasets\era5\pressure-005-002.nc")
rain = xr.open_dataset("Datasets\era5/rain-002-004.nc")
radiation = xr.open_dataset("Datasets\era5/radiation-004-003.nc")
temp = xr.open_dataset("Datasets\era5/temperature-003-005.nc")
biomass = xr.open_dataset("Datasets\era5/biomass.nc")
lightning = xr.open_dataset("Datasets\era5\lightning_1995-2011.nc")
pop = xr.open_dataset("Datasets\era5\population_1850-2100.nc")

# print(soilmoisture) #lon 464(-124.9 to -67.06), lat 224(25.06 to 52.94)
# print(wind.time) #lon 3600(0-360), lat 1801(90 to -90)
# print(lai.time) #lon 3600(0-360), lat 1801(90 to -90)
# print(pressure.time) #lon 3600(0-360), lat 1801(90 to -90)
# print(rain.time)#lon 3600(0-360), lat 1801(90 to -90)
# print(radiation.time)#lon 3600(0-360), lat 1801(90 to -90)
# print(temp.time)#lon 3600(0-360), lat 1801(90 to -90)
# print(biomass.time) #lon 720(-180-180), lat 360(90 to -90)
# print(lightning.time)#lon 192(0-360), lat 94(-90 to 90)
# print(pop.time) # lon 720 (-179.8-179.8), lat 360(-90 to 90)

wind = wind.assign_coords(longitude =(((wind.longitude + 180) % 360) - 180)).sortby('longitude')
lai = lai.assign_coords(longitude =(((lai.longitude + 180) % 360) - 180)).sortby('longitude')
pressure = pressure.assign_coords(longitude =(((pressure.longitude + 180) % 360) - 180)).sortby('longitude')
rain = rain.assign_coords(longitude =(((rain.longitude + 180) % 360) - 180)).sortby('longitude')
radiation = radiation.assign_coords(longitude =(((radiation.longitude + 180) % 360) - 180)).sortby('longitude')
temp = temp.assign_coords(longitude =(((temp.longitude + 180) % 360) - 180)).sortby('longitude')
lightning = lightning.assign_coords(lon =(((lightning.lon + 180) % 360) - 180)).sortby('lon')
soilmoisture = soilmoisture.reindex(lat=list(reversed(soilmoisture.lat)))
pop = pop.reindex(lat=list(reversed(pop.lat)))
lightning = lightning.reindex(lat=list(reversed(lightning.lat)))

#Making the Global Emission dataset into a USA only Dataset
datasets = [soilmoisture, wind, wind, lai, lai, pressure, rain, radiation, temp, biomass, lightning, pop]
varnames = ["SoilM_0_100cm", "u10", "v10", "lai_hv", 'lai_lv', "sp", "tp", "ssrd", "t2m", "biomass", "lnfm", "hdm"]
# for i in range(0, len(datasets)):
#     plt.matshow(datasets[i][varnames[i]][0])
#     plt.title(varnames[i])
# plt.show()

newnames = ["SoilM_0_100cmUSA", "u10USA", "v10USA", "lai_hvUSA", 'lai_lvUSA', "spUSA", "tpUSA", "ssrdUSA", "t2mUSA", "biomassUSA", "lnfmUSA", "hdmUSA"]
nonCAcoords = []

for index in range(0,1):
    file = datasets[index]
    #reducing the file to only coordinates in the bounding box of CA, bounding box nums found online
    fileUSbound = file[varnames[index]][:,:,:].where((file['lon']>=-125.0011) & (file['lon']<=-66.945392) & (file['lat']<=49.382808) & (file['lat']>=24.9493), drop=True)
    plt.matshow(fileUSbound[0])
    plt.title(varnames[index])
plt.show()
    # #looping through all values of time, lat, lon and replacing non-CA grid cells with nan(non-CA inside the bounding box)
    # for i in range(0, len(fileCAbound['time'])):
    #     if (i==0 and index == 0):
    #         for k in range(0, len(lat)):
    #             for j in range(0, len(lon)):
    #                 print(index, i, k, j)
    #                 #function call to detect what state a coordinate pair is in
    #                 result = us.from_coords(lat[k].values,lon[j].values)
    #                 #keeps track of non-CA grid cell indexes during first iteration
    #                 if (len(result)==0 or result[0].abbr != "CA"):
    #                     temp[i][k][j] = np.nan
    #                     nonCAcoords.append((k,j))
    #     else:
    #         #only loops through list of nonCA grid cells after first iteration and replaces with nan values
    #         for (lat1,lon1) in nonCAcoords:
    #             temp[i][lat1][lon1] = np.nan

    # x = newnames[index]
    # #creating dataset and saving as netcdf files
    # newfm = xr.Dataset(
    #     data_vars=dict(x = (['time', 'lat', 'lon'], temp.values)),
    #     coords=dict(lon=(temp.x.values), lat=(temp.y.values), time=(temp.time)),
    #     attrs= file.attrs
    # )
    # newfm.to_netcdf(path=(newnames[index] +'.nc'))





#Loading all CA datasets(All turned into CA datasets like the BA dataset)
# GEmCA = xr.open_dataset('Wildfirework/CAdatasets/GEmCA.nc')
# timestep = 1
# BACA = xr.open_dataset("Wildfirework/CAdatasets/BACAnew.nc")
# aspectCA = xr.open_dataset("Wildfirework/CAdatasets/aspectCA.nc")
# biCA = xr.open_dataset("Wildfirework/CAdatasets/biCA.nc")
# elevationCA = xr.open_dataset("Wildfirework/CAdatasets/elevationCA.nc")
# ercCA = xr.open_dataset("Wildfirework/CAdatasets/ercCA.nc")
# eviCA = xr.open_dataset("Wildfirework/CAdatasets/eviCA.nc")
# fm1000CA = xr.open_dataset("Wildfirework/CAdatasets/fm1000CA.nc")
# fm100CA = xr.open_dataset("Wildfirework/CAdatasets/fm100CA.nc")
# gdpCA = xr.open_dataset("Wildfirework/CAdatasets/gdpCA.nc")
# ltcCA = xr.open_dataset("Wildfirework/CAdatasets/ltcCA.nc")
# nppCA = xr.open_dataset("Wildfirework/CAdatasets/nppCA.nc")
# petCA = xr.open_dataset("Wildfirework/CAdatasets/petCA.nc")
# popCA = xr.open_dataset("Wildfirework/CAdatasets/popCA.nc")
# prCA = xr.open_dataset("Wildfirework/CAdatasets/prCA.nc")
# rmaxCA = xr.open_dataset("Wildfirework/CAdatasets/rmaxCA.nc")
# rminCA = xr.open_dataset("Wildfirework/CAdatasets/rminCA.nc")
# slopeCA = xr.open_dataset("Wildfirework/CAdatasets/slopeCA.nc")
# sphCA = xr.open_dataset("Wildfirework/CAdatasets/sphCA.nc")
# thCA = xr.open_dataset("Wildfirework/CAdatasets/thCA.nc")
# tmmnCA = xr.open_dataset("Wildfirework/CAdatasets/tmmnCA.nc")
# tmmxCA = xr.open_dataset("Wildfirework/CAdatasets/tmmxCA.nc")
# vpdCA = xr.open_dataset("Wildfirework/CAdatasets/vpdCA.nc")
# vsCA = xr.open_dataset("Wildfirework/CAdatasets/vsCA.nc")
# datasets = [biCA, gdpCA, nppCA, popCA, eviCA, ltcCA, aspectCA, elevationCA, slopeCA, vsCA, vpdCA, tmmxCA, tmmnCA, thCA, sphCA, rminCA, rmaxCA, prCA, petCA, fm1000CA, fm100CA, ercCA]
# varnames = ["biCA", "gdpCA", "nppCA", "popCA", "eviCA", "ltcCA", "aspectCA", "elevationCA", "slopeCA", "vsCA", "vpdCA", "tmmxCA", "tmmnCA", "thCA", "sphCA", "rminCA", "rmaxCA", "prCA", "petCA","fm1000CA", "fm100CA", "ercCA"]

    # temp = fileCAbound.rio.set_spatial_dims(x_dim='lon',y_dim='lat')
    # new = temp.rio.write_crs('epsg:4326')
    # upscale_factor = 1/2
    # new_width = int(new.rio.width * upscale_factor)
    # new_height = int(new.rio.height * upscale_factor)
    # downsampled = new.rio.reproject(
    #     new.rio.crs,
    #     shape=(new_height, new_width),
    #     resampling=Resampling.average)
    # lon = temp.x
    # lat = temp.y
    # temp = GEmCAbound.y.rio.set_spatial_dims(x_dim='lon',y_dim='lat')
# new = temp.rio.write_crs('epsg:4326')
# downsampledEM = new.rio.reproject(
#     new.rio.crs,
#     shape=(new_height, new_width),
#     resampling=Resampling.average)
# lon = temp.x
# lat = temp.y
# templat = []
# templon = []
# for lat in range(0, len(datasets[0]['lat'])):
#    temptemplat = []
#    temptemplon = []
#    for lon in range(0, len(datasets[0]['lon'])):
#       temptemplon.append(datasets[0]['lon'][lon].values)
#       temptemplat.append(datasets[0]['lat'][lat].values)
#    templat.append(temptemplat)
#    templon.append(temptemplon)   
      
# monthsin = []
# monthcos = []
# lat = []
# lon = []     

# for time in range(0, len(datasets[0]['time.month'])):
#    tempsin = np.empty((len(datasets[0]['lat']), len(datasets[0]['lon'])))
#    tempsin.fill(np.sin((2*np.pi*int(datasets[0]['time.month'][time].values))/12))
#    tempcos = np.empty((len(datasets[0]['lat']), len(datasets[0]['lon'])))
#    tempcos.fill(np.cos((2*np.pi*int(datasets[0]['time.month'][time].values))/12))
#    monthsin.append(tempsin)
#    monthcos.append(tempcos)
#    lat.append(templat)
#    lon.append(templon)

# monthsin, monthcos, lat, lon = np.array(monthsin), np.array(monthcos), np.array(lat), np.array(lon)

# arrs = []
# arrs.append(np.reshape(monthsin[:192], (-1, len(datasets[0]['lon']),1)))
# arrs.append(np.reshape(monthcos[:192], (-1, len(datasets[0]['lon']),1)))
# arrs.append(np.reshape(lat[:192], (-1, len(datasets[0]['lon']),1)))
# arrs.append(np.reshape(lon[:192], (-1, len(datasets[0]['lon']),1)))

# for i in range(0,len(datasets)):
#     arrs.append(np.reshape(datasets[i]['x'][:192].values, (-1, len(datasets[i]['lon']),1)))
# arrs = np.array(arrs)
# print(arrs.shape)
# print(GEmCA['y'].shape)
# # x = 
# # y = np.reshape(y, (-1, 3,1))
# final = np.concatenate(arrs, axis=2)
# final = np.reshape(final, (-1, len(datasets[0]['lat']), len(datasets[0]['lon']), len(arrs)))
# print(datasets[1])
# print(final.shape)

# variables = []
# variables.append('month sin')
# variables.append('month cos')
# variables.append('lat')
# variables.append('lon')
# for i in range(0, len(datasets)):
#   variables.append(varnames[i])
# CAdata = xr.Dataset(
#     data_vars=dict(data=(['time','lat', 'lon', 'variables'], final), DM=(['time', 'lat','lon'], GEmCA), timestep=(timestep)),
#     coords=dict(time=(GEmCA.time.values), lat=(datasets[0]['lat'].values), lon=(datasets[0]['lon'].values),variables=(variables))
# )
# CAdata.to_netcdf(path='./CAEmdata_gridded(2x2_aggregated_new).nc')