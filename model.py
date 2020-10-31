#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 11:07:30 2020

@author: Sam Wondim
"""
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab
import mysql.connector as sql
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from feature_engine.categorical_encoders import MeanCategoricalEncoder, OneHotCategoricalEncoder
from feature_engine import missing_data_imputers as mdi
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials


import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.5f}'.format


raw_data = pd.read_csv('Data/house_data_details_eda.csv')
df = raw_data.copy()
df[['month', 'year']] = df[['month', 'year']].astype(str)
gdf = gpd.GeoDataFrame(df, geometry=[Point(xy)
                                     for xy in zip(df.longitude, df.latitude)])
gdf.crs = {'init': 'epsg:4326'}

# Cluster the properties by location

ssd = []

for i in range(2, 50):

  kmeans = KMeans(n_clusters=i, init='k-means++')
  kmeans.fit_predict(gdf[['latitude', 'longitude']])
  ssd.append(ssd.append(kmeans.inertia_))

ssd = ssd[::2]


# Elbow method to determine the optimal number for k
fig, ax = plt.subplots(figsize=(5, 5))

ax.plot(list(range(2, 50)), ssd, marker='o', markevery=3)
plt.title('Elbow Method')
plt.xlabel('K')
plt.ylabel('Sum of Square Errors')
ax.set(xticks=range(2, 50, 3))

# 8 seems like the optimal number of clusters
kmeans = KMeans(n_clusters=8, init='k-means++')
gdf['geolocation_cluster'] = kmeans.fit_predict(gdf[['latitude', 'longitude']])

# Plot the data point on a map of Los Angeles
fig, ax = plt.subplots(figsize=(10, 10))
# ax.set_aspect('equal')

la_full = gpd.read_file('LACounty/l.a. county neighborhood (v6).shp')
la_full.plot(ax=ax, alpha=0.4, edgecolor='darkgrey',
             color='lightgrey', aspect=1, zorder=1)

# Conver x_train df into geopandas df so that we can visualize it
temp_plot = gpd.GeoDataFrame(gdf.copy(), geometry=[Point(
    xy) for xy in zip(gdf.longitude, gdf.latitude)])
temp_plot.plot(ax=ax, aspect=1,
               c=temp_plot['geolocation_cluster'], alpha=0.2, linewidth=0.8, zorder=2)

kc = kmeans.cluster_centers_

for i in range(len(kc)):
  gdf['distance_' + str(i)] = np.sqrt((gdf.latitude - kc[i][0])
                                      ** 2 + (gdf.longitude - kc[i][1])**2)

# Drop the latitude and longitude
gdf.drop(['latitude', 'longitude'], axis=1, inplace=True)

targets = gdf['sale_price']
inputs = gdf.drop(
    ['sale_price', 'geometry', 'geolocation_cluster', 'month_year'], axis=1)

# Split the dataset into train and test set
x_train, x_test, y_train, y_test = train_test_split(
    inputs, targets, test_size=.2, random_state=24)


# Dummy encode the categorical variables
ohe_encoder = OneHotCategoricalEncoder(top_categories=7,
                                       variables=['property_type', 'crime_index', 'school_quality', 'month', 'year'])

ohe_encoder.fit(x_train)

x_train = ohe_encoder.transform(x_train)
x_test = ohe_encoder.transform(x_test)


# Optimize the objective function with hyperopt
def acc_model(params):
  est = int(params['n_estimators'])
  feat = int(params['max_features'])
  leaf = int(params['min_samples_leaf'])
  rf = RandomForestRegressor(
      n_estimators=est, max_features=feat, min_samples_leaf=leaf)
  return cross_val_score(rf, x_train, y_train).mean()


# Define the search space
param_space = {

    'n_estimators': hp.quniform('n_estimators', 25, 500, 1),
    'max_features': hp.quniform('max_features', 1, 40, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 3, 1)
}

best = 0


def f(params):
  global best
  acc = acc_model(params)
  if acc > best:
    best = acc
    print('new best:', best, params)
  return {'loss': -acc, 'status': STATUS_OK}


trials = Trials()
best_params = fmin(f, param_space, algo=tpe.suggest,
                   max_evals=25, trials=trials)
print('best:')
print(best_params)


rf = RandomForestRegressor(n_estimators=287,
                           max_features=21,
                           min_samples_leaf=2,
                           random_state=24)

rf.fit(x_train, y_train)
predictions = rf.predict(x_test)


# Use this code when using the log transformer
# Need to back transform the target variable

'''
predictions = rf.predict(x_test)
predictions = pd.Series([math.exp(x) for x in predictions])
y_test = pd.Series([math.exp(x) for x in y_test])

'''

score = np.sqrt(mean_squared_error(y_test, predictions))
print(f'Random Forest Score rmse: {score}')
