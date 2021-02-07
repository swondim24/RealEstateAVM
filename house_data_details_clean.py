#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 13:36:13 2021

@author: Samuel Wondim
"""


#Import package(s)
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import details
from shapely.geometry import Point, MultiPolygon, Polygon
import numpy as np
from feature_engine import variable_transformers as vt
from feature_engine.outlier_removers import Winsorizer
import mysql.connector

#raw_data = pd.read_csv('Data/house_data_details_scraped.csv')
raw_data = pd.read_csv('Data/temp_house_data_details_scraped.csv')
df = raw_data.copy()

#Parse the bedroom feature
df['bedrooms'] = df['rooms'].apply(lambda x: x.split(',')[0] if x!=None and type(x)== str else None)
df['bedrooms'] = df['bedrooms'].replace('\D', '', regex=True)

#Parse the bathrooms feature
df['bathrooms'] = df['rooms'].apply(lambda x: x.split(',')[1] if x!=None and type(x)== str and x != 'Built in' else None)
df['bathrooms'] = df['bathrooms'].replace('\D', '', regex=True)

#Drop the Rooms column
df.drop('rooms', inplace=True, axis=1)

#Parse the home unit
df['home_unit'] = df['home_size'].apply(lambda x: x.split(' ')[1] if x!=None and type(x)==str else None)

#Parse the home size
df['home_size_temp'] = df['home_size'].apply(lambda x: x if x!=None else None)
df.loc[df['home_unit']=='acres', ['home_size_temp']] = df['home_size'].apply(lambda x: x.split(' ')[0] if x!=None and type(x)== str else None).replace('\D', '', regex=True)
df.loc[df['home_unit']=='sqft', ['home_size_temp']] = df['home_size'].apply(lambda x: x.split(' ')[0] if x!=None and type(x)== str else None).replace('\D', '', regex=True)
df['home_size'] = df['home_size_temp']

#Drop the temp column
df.drop(['home_size_temp'], inplace=True, axis=1)

#Parse the lot unit
df['lot_unit'] = df['lot_size'].apply(lambda x: x.split(' ')[1] if x!=None and type(x)==str else None)

#Parse the lot size
df['lot_size_temp'] = df['lot_size'].apply(lambda x: x if x!=None else None)
df.loc[df['lot_unit']=='acres', ['lot_size_temp']] = df['lot_size'].apply(lambda x: x.split(' ')[0] if x!=None and type(x)== str else None).replace('\D', '', regex=True)
df.loc[df['lot_unit']=='sqft', ['lot_size_temp']] = df['lot_size'].apply(lambda x: x.split(' ')[0] if x!=None and type(x)== str else None).replace('\D', '', regex=True)
df['lot_size'] = df['lot_size_temp']

#Drop the temp and unit columns
df.drop(['lot_size_temp'], inplace=True, axis=1)

#Convert the acres into sqft
#First convert the data type to float
df[['home_size', 'lot_size']] = df[['home_size', 'lot_size']].astype(float)

df.loc[df['home_unit']=='acres', 'home_size'] = df['home_size']*43560.0
df.loc[df['lot_unit']=='acres', 'lot_size'] = df['lot_size']*43560.0

#Drop the unit columns
df.drop(['home_unit', 'lot_unit'], inplace=True, axis=1)

#Parse and clean Zoning
df['zoning'] = df['zoning'].replace('\s', '', regex=True).replace('Zoning:', '', regex=True).replace('\*?', '', regex=True)

#Parse out the sale price and estimated value
df['sale_price'] = df['sale_price'].replace('\D', '', regex=True)
df['estimated_value'] = df['estimated_value'].replace('\D', '', regex=True)

#Parse out sex_offenders, enviornmental_hazards, natural_disasters
df[['sex_offenders', 'enviornmental_hazards', 'natural_disasters']] = df[['sex_offenders', 'enviornmental_hazards', 'natural_disasters']].replace('\D', '', regex=True)

#All of the features that are to be numeric
numeric = ['sale_price', 'estimated_value', 'bedrooms', 'bathrooms', 'sex_offenders',  
           'enviornmental_hazards', 'natural_disasters']
date = ['date']
objects = ['parcel_number', 'realtyID']

#Convert home_size and lot_size to numeric types
df[numeric] = df[numeric].apply(pd.to_numeric)

#Convert the date of the sale price to date_time
df[date] = df[date].apply(pd.to_datetime)

#Convert the parcel_number and realtyid features into strings
df[objects] = df[objects].astype(str)

#Fix the geolocations for those that are reversed
proper_lat = df.loc[df['latitude']<33, 'longitude']
proper_long = df.loc[df['longitude']>-116, 'latitude']

df.loc[df['latitude']<33, 'latitude'] = proper_lat
df.loc[df['longitude']>-116, 'longitude'] = proper_long

#Remove any properties in the Catalina Island
df = df.loc[df['latitude']>33.6]

#Remove any duplicate houses
df = df.drop_duplicates(subset='parcel_number')

gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.longitude, df.latitude)])

#Import geojson map of los angeles
hoods = gpd.read_file('Los Angeles Neighborhood Map.geojson')
hoods = hoods.rename(columns={'name': 'neighborhood',
                              'latitude': 'longitude',
                              'longitude': 'latitude'})
hoods[['latitude', 'longitude']] = hoods[['latitude', 'longitude']].astype(float)
hoods['polygon'] = hoods['geometry'].apply(lambda x: Polygon(x[0]))

#This function will classify every house and label it an appropriate neighborhood
def find_neighborhood(coordinates):
    for index, hood in hoods.iterrows():
        if hood['polygon'].contains(coordinates):
            return hood['neighborhood']
        else:
            continue 

gdf['neighborhood'] = gdf['geometry'].apply(find_neighborhood)

#Code below will assign the nearest neighborhood for those with a missing value
for index, row in gdf.iterrows():
    if row['neighborhood'] == None:
        lat = row['latitude']
        lng = row['longitude']
        hoods2 = hoods.copy()
        hoods2['distance'] = np.sqrt((hoods2['latitude'] - lat)**2 + (hoods2['longitude'] - lng)**2)
        hoods2 = hoods2.sort_values(by='distance').reset_index(drop=True)
        gdf.loc[index, 'neighborhood'] = hoods2['neighborhood'][0]
        
#Before removing the observations with missing values, 
#let's remove the features we won't need so we can limit the number of observations we will need
dropped_features = ['realtyID', 'county', 'subdivision', 'census', 'tract', 'lot', 'estimated_value']

#Remove missing values for the target feature
num_obs_1 = df.shape[0]
print(f'{num_obs_1} number of observations before trim')

df.drop(dropped_features, axis=1, inplace=True)
df.dropna(inplace=True)
num_obs_2 = num_obs_1 - df.shape[0]
num_obs_3 = num_obs_1 - raw_data.dropna().shape[0]
#print(f'{num_obs_3} number of missing observations when including the bad features')
#print(f'{num_obs_2} number of missing observations when not including the bad features')
#print(f'{df.shape[0]} current number of observations')

#Convert date into a datetime object
df['date'] = pd.to_datetime(df['date'])
df.sort_values(by='date', ascending=False, inplace=True, ignore_index=True)

#Correct the dates
df['sold_date'] = df['date'].apply(lambda x: x.replace(year= x.year-100) if x.year>2021 else x)
df['date'] = df['sold_date'].values.astype('datetime64[M]') #Converting the date column into the first of the month
df.sort_values(by='date', ascending=False, inplace=True, ignore_index=True)
df.drop_duplicates(inplace=True)

#We will use the cpi for every corresponding sale date to help adjust the sale price based on inflation
#Add a new column that will be used for the merge

df['year'] = df['date'].dt.year

#Read in a file with consumer price index
cpi_df = pd.read_csv('Data/CPIAUCNS.csv')
cpi_df.rename(columns={'DATE':'date', 'CPIAUCNS':'cpi'}, inplace=True)
cpi_df['date'] = pd.to_datetime(cpi_df['date'])
cpi_df['date'] = cpi_df['date'].apply(lambda x: x.replace(year= x.year-100) if x.year>2021 else x)
cpi_df.sort_values(by='date', ascending=False, inplace=True, ignore_index=True)

merged_df = pd.merge(df, cpi_df, on='date', how='left')
merged_df.drop_duplicates(inplace=True)

cpi_2021 = int(merged_df.loc[0, ['cpi']])
merged_df['sale_price_cpi'] = (merged_df['sale_price']*cpi_2021)/ merged_df['cpi']

#Define a function that calculates the boundaries for a feature
def find_skewed_boundaries(df, var, distance):
    
    IQR = df[var].quantile(.75) - df[var].quantile(.25)
    
    lower_boundary = df[var].quantile(.25) - (distance*IQR)
    upper_boundary = df[var].quantile(.75) + (distance*IQR)
    
    return upper_boundary, lower_boundary

sales_upper_boundary, sales_lower_boundary = find_skewed_boundaries(merged_df, 'sale_price_cpi', 1.0)


#Remove outlier values
obs1 = merged_df.shape[0]
merged_df = merged_df.loc[merged_df['sale_price_cpi']<sales_upper_boundary,:]
merged_df = merged_df.loc[merged_df['sale_price_cpi']>75000]
obs2 = merged_df.shape[0]

#This function will transform variables into US Currency
prop_func = lambda x: '${:,.2f}'.format(x)

#Plot for the sale_price distribution
chart = sns.displot(merged_df['sale_price_cpi'], bins=50)
plt.title('Sale Price with outliers removed')
plt.ticklabel_format(style='plain', axis='x')
skip_step = (merged_df['sale_price_cpi'].max() - merged_df['sale_price_cpi'].min()) / 10
labels = np.round(np.arange(merged_df['sale_price_cpi'].min(), merged_df['sale_price_cpi'].max(), skip_step-1),2)
prop_labels = list(map(prop_func, labels))
chart.set(xticks=labels)
chart.set(xticklabels=prop_labels)
chart.set_xticklabels(rotation=25);

#Power Transformer
et = vt.PowerTransformer(variables = ['sale_price_cpi', 'sale_price'])
et.fit(merged_df)

power_df = et.transform(merged_df)

sales_upper_boundary, sales_lower_boundary = find_skewed_boundaries(power_df, 'sale_price_cpi', 1.0)


#Remove outlier values
obs1 = power_df.shape[0]
#print(f'Number of observations before removing outliers: {obs1}')
power_df = power_df.loc[power_df['sale_price_cpi']<sales_upper_boundary,:]
power_df = power_df.loc[power_df['sale_price_cpi']>sales_lower_boundary]
obs2 = power_df.shape[0]
#print(f'Number of observations after removing outliers: {obs2}')
print(f'Removed {obs1-obs2} observations that were above the sales price boundary')


#Below you will see the change in price when adjusting for inflation
fig = plt.figure()
real_avg = power_df.groupby('year')['sale_price'].mean().plot(label='Original Sale Price')
adj_avg = power_df.groupby('year')['sale_price_cpi'].mean().plot(label='Sale Price Adjusted for Inflation')
plt.legend()
#plt.savefig('Images/sale_price_adjusted.png')

#Determine the age of the property 
power_df['age'] = power_df['date'].dt.year - power_df['year_built']

#Remove every property type except Single Family Residence 
num_1 = power_df.shape[0]
removed_df = power_df.loc[power_df['property_type']!='Single Family Residence']
num_2 = removed_df.shape[0]
power_df = power_df.loc[power_df['property_type']=='Single Family Residence']
print(f'{power_df.shape[0]} number of Single Family Residences remain')

#Remove outliers for lot size
lot_size_upper_boundary, lot_size_lower_boundary = find_skewed_boundaries(power_df, 'lot_size', 3.0)


#Remove outlier values
trimmed_df = power_df.loc[power_df['lot_size']<lot_size_upper_boundary,:]

print(f'Removed {power_df.shape[0]-trimmed_df.shape[0]} observations due to lot size')

#Cap the outliers for sex offenders and enviornmental hazards features
windsoriser = Winsorizer(distribution='skewed',
                          tail='right', 
                          fold=1.5,
                          variables=['sex_offenders','enviornmental_hazards'])

windsoriser.fit(trimmed_df)

trimmed_df = windsoriser.transform(trimmed_df)

#Create a dataset with just unique houses
drop_features = ['property_type', 'url', 'year', 'cpi', 'natural_disasters', 'zoning', 'geometry', 'date']
houses = trimmed_df.sort_values(by='sold_date', ascending=False).reset_index(drop=True)
print(f'Number of observations before removing duplicate houses {houses.shape[0]}')
houses = houses.drop_duplicates(subset='parcel_number')
houses.drop(drop_features, axis=1, inplace=True)
houses.rename(columns={'sold_date': 'date'}, inplace=True)
houses['sale_price'] = houses['sale_price']**2
houses[['age', 'bedrooms', 'bathrooms', 'sex_offenders', 'enviornmental_hazards', 'year_built']] = houses[['age', 'bedrooms', 'bathrooms', 'sex_offenders', 'enviornmental_hazards', 'year_built']].astype(int)
houses['parcel_number'] = houses['parcel_number'].astype(float)
houses['parcel_number'] = houses['parcel_number'].astype(int)
houses['parcel_number'] = houses['parcel_number'].astype('O')
print(f'Number of observations after removing duplicate houses {houses.shape[0]}')
#houses.to_csv('Data/houses.csv', index=False)

sql_order = ['parcel_number', 'address', 'latitude', 'longitude', 'home_size', 'lot_size', 'year_built', 'age', 'sex_offenders', 'crime_index', 'enviornmental_hazards', 'school_quality', 'bedrooms', 'bathrooms', 'neighborhood', 'sale_price', 'sale_price_cpi', 'date' ]
houses = houses[sql_order]


db = mysql.connector.connect(
    
    host = 'Samuels-MacBook-Air.local',
    user = 'root',
    passwd = details.db_password,
    database = 'realestate_avm'
)

mycursor = db.cursor()

for index, row in houses.iterrows():
    try:
        mycursor.execute('INSERT INTO Homes(ParcelNumber, Address, Latitude, Longitude, HomeSize, LotSize, YearBuilt, Age, SexOffenders, CrimeIndex, EnviornmentalHazards, SchoolQuality, Bedrooms, Bathrooms, Neighborhood, SalePrice, SalePriceTransformed, Date) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, %s)', tuple(row))
    except:
        continue
db.commit()

mycursor.close()
db.close()

