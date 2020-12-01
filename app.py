import streamlit as st
import pandas as pd
import folium
from folium import plugins
from streamlit_folium import folium_static
import time
from functions import find_nearest_properties


st.title("How much is it really going to cost to buy a house in Los Angeles?")

st.sidebar.write('<strong>Describe your ideal house</strong>',
                 unsafe_allow_html=True)

# Load in the data
neighborhoods = pd.read_csv('Data/Neighborhoods_final.csv')
houses = pd.read_csv('Data/houses.csv')

# Add an option for search method
search = st.sidebar.radio('', ('Basic Search', 'Advanced Search'))
if search == 'Basic Search':
  los_angeles = [34.052235, -118.243683]
  la = folium.Map(location=los_angeles)
  # Add markers to the map for each unique neighborhood
  mc = plugins.MarkerCluster()
  for index, row in neighborhoods.iterrows():
    popup = row['neighborhood'] + '<br>' 'Average Price: ' + row['sale_price']
    mc.add_child(folium.Marker(location=row[['longitude', 'latitude']],
                               # tooltip=row['neighborhood'],
                               popup=popup,
                               cluster_marker=True))
  la.add_child(mc)
  folium_static(la)

else:
  neighborhood_option = st.sidebar.selectbox(
      'Select a neighborhood you are interested in.',
      neighborhoods['neighborhood'])
  # Creating an side bar for the features
  left_column, right_column = st.sidebar.beta_columns(2)
  bedrooms = left_column.number_input(
      label="Bedrooms", format="%i", value=2, max_value=8, min_value=1)
  bathrooms = right_column.number_input(
      label="Bathrooms", format="%i", value=2, max_value=8, min_value=1)
  lot_size = st.sidebar.number_input(
      label="Lot Size (You can also type in a numbers)", format="%i", value=1000, step=50)
  home_size = st.sidebar.number_input(
      label="Home Size (You can also type in a number)", format="%i", value=1000, step=50)
  search = st.sidebar.button('Search')

  if search:
    find_nearest_properties(neighborhood_option, bedrooms, bathrooms, lot_size, home_size)
