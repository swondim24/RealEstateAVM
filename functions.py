# This module will contain all the functions requried for app.py
import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.cluster import KMeans
import folium
from folium import plugins
from streamlit_folium import folium_static
import time
import pickle


# This function will allow us to load our models

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# This function calculates the distances between the cluster centers


def calculate_distances(df, kc):
    for i in range(len(kc)):
        df['distance_' + str(i)] = (df.latitude -
                                    kc[i][0])**2 + (df.longitude - kc[i][1])**2
    df = df.drop(['longitude', 'latitude'], axis=1)
    return df


def search_function(map_filter, neighborhood_option):
    la = folium.Map(
        location=map_filter, popup=neighborhood_option, zoom_start=12, tiles=None, collapsed=False)

    # folium.GeoJson('Los Angeles Neighborhood Map.geojson',
    #               name='Neighborhood Filter').add_to(la)

    folium.raster_layers.TileLayer(
        tiles='openstreetmap', name='Street Map').add_to(la)

    folium.LayerControl().add_to(la)

    folium_static(la)


def first():
    st.title("How much is it really going to cost to buy a house in Los Angeles?")

    st.sidebar.write('<strong>Describe your ideal house</strong>',
                     unsafe_allow_html=True)

    # Load in the data
    neighborhoods = pd.read_csv('Data/la_neighborhoods.csv')
    #los_angeles = [34.052235, -118.243683]
    #la = folium.Map(location=los_angeles)

    neighborhood_option = st.selectbox(
        'Select a neighborhood you are interested in.',
        neighborhoods['neighborhood'])

    los_angeles = [34.052235, -118.243683]
    la = folium.Map(location=los_angeles)

    map_filter = neighborhoods.loc[neighborhoods['neighborhood']
                                   == neighborhood_option, ['longitude', 'latitude']]

    left_column, right_column = st.sidebar.beta_columns(2)

    bedrooms = left_column.slider('Bedrooms', 0, 8, 2)
    bathrooms = right_column.slider('Bathrooms', 0, 8, 2)

    search = st.sidebar.button('Search')

    if search:
        la = functions.search_function(map_filter, neighborhood_option)
        st.write(f'Finding houses near {neighborhood_option}')
        # folium_static(la)

    folium_static(la)


def find_nearest_hoods(neighborhood_option):
    hoods = pd.read_csv('Data/Neighborhoods_final.csv')
    lat = float(hoods.loc[hoods['neighborhood'] ==
                          neighborhood_option, 'latitude'])
    lng = float(hoods.loc[hoods['neighborhood'] ==
                          neighborhood_option, 'longitude'])

    hoods['distance'] = np.sqrt(
        (hoods.latitude - lat)**2 + (hoods.longitude - lng)**2)

    hoods.sort_values(by='distance', inplace=True)

    return list(hoods['neighborhood'][:8])


def find_similar_properties(neighborhood_option, bedrooms, bathrooms, num=5.0):
    neighborhoods = pd.read_csv('Data/Neighborhoods_final.csv')
    houses = pd.read_csv('Data/houses_neighborhood_info.csv')
    near = find_nearest_hoods(neighborhood_option)
    hood_lat = float(neighborhoods.loc[neighborhoods['neighborhood'] ==
                                       neighborhood_option, 'latitude'])
    hood_lng = float(neighborhoods.loc[neighborhoods['neighborhood'] ==
                                       neighborhood_option, 'longitude'])
    houses = houses[houses.neighborhood.isin(near)]
    clusters = int(float(houses.shape[0]) / num)
    kmeans = KMeans(n_clusters=clusters)
    houses['cluster'] = kmeans.fit_predict(
        houses[['latitude', 'longitude', 'bedrooms', 'bathrooms']])

    pred = int(kmeans.predict(
        np.array([[hood_lat, hood_lng, bedrooms, bathrooms]])))

    houses = houses.loc[houses['cluster'] == pred, :]

    return houses


def find_nearest_properties(neighborhood_option, bedrooms=2, bathrooms=2, home_size=1000, lot_size=1000):

    neighborhoods = pd.read_csv('Data/Neighborhoods_final.csv')
    # load the model
    loaded_model = pickle.load(open("model.pickle.dat", "rb"))
    cols_when_model_builds = loaded_model.get_booster().feature_names
    # Load the fitted kmeans object
    kmeans = load_obj('kmeans_neighborhood')
    kc = kmeans.cluster_centers_
    # This  is the geocoordinates of the neighborhood selected
    map_filter = neighborhoods.loc[neighborhoods['neighborhood']
                                   == neighborhood_option]
    map_filter.loc[0, ['home_size', 'lot_size', 'bedrooms', 'bathrooms']] = [
        home_size, lot_size, bedrooms, bathrooms]
    lng = map_filter.longitude
    lat = map_filter.latitude
    map_filter = calculate_distances(map_filter, kc)
    map_filter.drop(['neighborhood', 'sale_price'], axis=1, inplace=True)
    map_filter = map_filter[cols_when_model_builds]
    house_params = map_filter.astype(float)

    est_price = int(np.exp(loaded_model.predict(house_params)))
    clean_price = '${:,.2f}'.format(est_price)
    # Bring in similar houses
    houses = find_similar_properties(neighborhood_option, bedrooms, bathrooms)

    if houses.shape[0] == 0:
        st.header(f'We cannot find any houses matching that description.')
    else:
        results = f"Estimated price: **{clean_price}** "
        mes1 = f"We have found {houses.shape[0]} houses with a similar description near {neighborhood_option}*(You may have to zoom out to see all the houses)*"
        st.markdown(results)
        st.markdown(mes1)

    houses['date'] = houses['date'].apply(
        lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').strftime('%m/%d/%Y'))
    houses['sale_price'] = houses['sale_price'].apply(
        lambda x: '${:,.2f}'.format(x))
    la = folium.Map(location=[lng, lat],
                    zoom_start=13,
                    min_zoom=8)
    folium.Marker(location=[lng, lat],
                  popup=neighborhood_option).add_to(la)

    for index, row in houses.iterrows():
        popup = 'Sale Price: ' + \
            str(row['sale_price']) + '<br>' + 'Sell Date: ' + row['date'] + '<br > ' + \
            'Neighborhood: ' + row['neighborhood'] + '<br >' + \
            'Bedrooms: ' + str(int(row['bedrooms'])) + '<br >' + \
            'Bathrooms: ' + str(int(row['bathrooms'])) + '<br>' +\
            'Home Size: ' + str(int(row['home_size'])) + ' sqft' + '<br >' + \
            'Lot Size: ' + str(int(row['lot_size'])) + ' sqft'
        folium.Marker(location=row[['latitude', 'longitude']],
                      popup=popup,
                      icon=folium.Icon(color='lightgray', icon='home', ico_size=(3, 3))).add_to(la)

    folium_static(la)
