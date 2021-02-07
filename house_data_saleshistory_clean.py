#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 16:03:01 2021

@author: Samuel Wondim
"""


import pandas as pd
import mysql.connector as sql
import details

df = pd.read_csv('Data/temp_house_data_saleshistory_scraped.csv')

#Remove all of the extra white space for date
df['date'] = df['date'].replace('\s', '', regex=True)

#Keep all of the digits for price and price_sqft
df['price'] = df['price'].replace('\D', '', regex=True)
df.head()

df['parcel_number'] = df['parcel_number'].map(str)
df['date'] = pd.to_datetime(df['date'])
df['price'] = pd.to_numeric(df['price'])

#Remove duplicates
df = df.drop_duplicates()

#Remove observations with missing values
df = df.dropna()

#Upload the saleshistory data to mysql database
db_connection = sql.connect(
    
    host = 'Samuels-MacBook-Air.local',
    user = 'root',
    passwd = details.db_password,
    database = 'realestate_avm'
)

#Load the unique houses from the database
homes = pd.read_sql('SELECT * FROM Homes;', con=db_connection)
parcel = list(homes['ParcelNumber'])

mycursor = db_connection.cursor()

query = 'INSERT INTO SalesHistory (ParcelNumber, Date, SalePrice) VALUES(%s, %s,%s)'                                                         

my_data = []
for index, row in df.iterrows():
    #This is to ensure that we only enter records that have a home in the Homes table
    if row['parcel_number'] in parcel:
        my_data.append(tuple(row))
    else:
        continue

mycursor.executemany(query, my_data)

db_connection.commit()
mycursor.close()

db_connection.close()

print('Done!')