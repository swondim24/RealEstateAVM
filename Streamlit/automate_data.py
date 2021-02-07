#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 18:47:52 2021

@author: Samuel Wondim
"""
import pandas as pd
import geopandas as gpd
import mysql.connector as sql
import details
from feature_engine.categorical_encoders import MeanCategoricalEncoder, OneHotCategoricalEncoder


#Load the data from mysql
password = details.db_password
db_connection = sql.connect(host='Samuels-MacBook-Air.local', 
                            database='realestate_AVM',
                            user='root',
                            password=password)

homes = pd.read_sql('SELECT * FROM Homes;', con=db_connection)
homes.drop(['ParcelNumber', 'Address'], axis=1, inplace=True)


#Save the first dataframe that we will need for the application
homes.to_csv('Data/houses.csv', index=False)

#Create a copy of the homes dataframe
neighborhoods = homes.copy()

#We are going to create another column that stores the mean target value for each neighborhood
#First we need to save the neighborhood name into another column
neighborhoods['neighborhood_name'] = neighborhoods['Neighborhood']

mean_enc = MeanCategoricalEncoder(variables=['Neighborhood'])

#Fit the encoder
mean_enc.fit(neighborhoods, neighborhoods['SalePrice'])

#Transform the neighborhoods dataframe
neighborhoods = mean_enc.transform(neighborhoods)

#Load a dataframe that has the geocoordinates of each neighborhood
hoods = gpd.read_file('Los Angeles Neighborhood Map.geojson')
hoods = hoods.rename(columns={'name': 'neighborhood_name'})
hoods[['longitude', 'latitude']] = hoods[['longitude', 'latitude']].astype(float)

#Create a dataframe that averages out every attribute by neighborhood
avg = neighborhoods.groupby('neighborhood_name')[['HomeSize', 'LotSize', 'Bedrooms', 'Bathrooms', 'SexOffenders', 
                                   'EnviornmentalHazards', 'Age', 'SalePrice', 'Neighborhood', 'YearBuilt']].mean()

#Do the same thing as above but for only categorical variables
cat_avg = neighborhoods.groupby('neighborhood_name')[['SchoolQuality', 'CrimeIndex']].agg(lambda x:x.value_counts().index[0])

#Create a new column that has sale price written in proper form
avg['SalePrice'] = avg['SalePrice'].apply(lambda x: '${:,.2f}'.format(x))

#Merge the first dataframe with the dataframe that includes the geolocation attributes
new_avg = pd.merge(avg, hoods[['neighborhood_name', 'latitude', 'longitude']], 
                   on ='neighborhood_name',  how='left')

#Merge the previously formed dataframe with the categorical dataframe
new_avg = pd.merge(new_avg, cat_avg, on='neighborhood_name', how='left')

#Remove observations with missing values
new_avg = new_avg.dropna()

#One Hot encode the categorical variables in order to prepare it for the app
ohe_encoder = OneHotCategoricalEncoder(variables=['CrimeIndex', 'SchoolQuality'])
ohe_encoder.fit(new_avg)
new_avg = ohe_encoder.transform(new_avg)

#Accounting for a rare label
if 'CrimeIndex_Very High' not in new_avg.columns:
    new_avg['CrimeIndex_Very High'] = 0

#Save the Neighborhood DataFrame    
new_avg.to_csv('Data/Neighborhoods_final.csv', index=False)