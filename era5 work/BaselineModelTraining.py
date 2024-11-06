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
from sklearn.utils import resample
import math
import sklearn as sk
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import smogn
import rioxarray as rio
import copy
import seaborn as sns
import tables
import h5py
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

dataset = pd.read_csv("Datasets\era5\TrainingData.csv")
# dataset = pd.read_csv("Datasets\era5\TrainingData.csv")

mse = tf.keras.losses.MeanSquaredError()
sc = MinMaxScaler()

X = dataset.drop(columns='Next_DM')
Y = dataset['Next_DM']
Y = np.log(Y).replace(-np.inf, 0)
Y = Y +np.abs(np.min(Y))


fire_X_train, fire_X_test, fire_y_train, fire_y_test = train_test_split(X, Y, test_size = 0.25, random_state=0)
fire_X_train = sc.fit_transform(fire_X_train)
fire_X_test = sc.transform(fire_X_test)
print(fire_X_test)
print(fire_y_test)

model = Sequential([
    Dense(128, input_shape=(len(fire_X_train[0]),), activation='relu'),
    Dense(128, activation = 'relu'),
    Dense(64, activation = 'relu'),
    Dense(64, activation = 'relu'),
    Dense(64, activation = 'relu'),
    Dense(32,activation='relu'),
    Dense(1, activation='relu')
])

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
FC = model.fit(fire_X_train, fire_y_train, epochs=50, batch_size=32, validation_split=0.3)
pred = model.predict(fire_X_test)

f = plt.figure()
f.set_figwidth(25)
f.set_figheight(12)
plt.title("Learning Curves")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.plot(FC.history['loss'], label='train')
plt.plot(FC.history['val_loss'], label='val')
plt.grid()
plt.legend()
plt.show()

metric = tf.keras.metrics.R2Score()
metric.update_state(fire_y_test, pred)
result = metric.result()
result.numpy()
print("R^2 for test dataset: " + str(result))
m = tf.keras.metrics.MeanSquaredError()
m.update_state(fire_y_test, pred)
result = m.result().numpy()
print("MSE: "+str(result))

plt.scatter(x = pred, y = fire_y_test)
plt.xlabel("Pred")
plt.ylabel("True")
plt.xlim(np.min(fire_y_test), np.max(fire_y_test))
plt.ylim(np.min(fire_y_test), np.max(fire_y_test))
plt.show()