import streamlit as st
import pandas as pd
import folium
from folium import plugins
from streamlit_folium import folium_static
import time
from functions import find_nearest_properties, calculate_mortgage, interest_bar_graph, amortization_schedule
st.set_option('deprecation.showPyplotGlobalUse', False)

import warnings
warnings.filterwarnings('ignore')


st.title("How much is it really going to cost to buy a house in Los Angeles?")

st.sidebar.write('<strong>Describe your ideal house</strong>',
                 unsafe_allow_html=True)

# Load in the data
neighborhoods = pd.read_csv('Data/Neighborhoods_final.csv')
houses = pd.read_csv('Data/houses.csv')


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
    label="Lot Size (You can also type in a number)", format="%i", value=6500, step=250, max_value=25000)
home_size = st.sidebar.number_input(
    label="Home Size (You can also type in a number)", format="%i", value=1500, step=250, max_value=25000)
#search = st.sidebar.button('Search')
est_price = find_nearest_properties(neighborhood_option, bedrooms,
                                    bathrooms, lot_size, home_size)
value = int(est_price * .2)

left, middle, right = st.beta_columns(3)
down_payment = left.number_input(
    label="Down Payment", format="%i", value=20000, max_value=5000000, step=1000)
loan_term = middle.selectbox(
    "Loan Term",  [15, 30], 1)
interest_rate = right.slider(
    label="Interest Rate",  value=2.5, max_value=8.0, step=.05)

payment = calculate_mortgage(
    est_price, down_payment, loan_term, interest_rate)

mes1 = "*You\'re interest rate will be lower the less years there are on the term*"
st.markdown(mes1)
mes2 = f"## You're expected monthly payment is {'${:,.2f}'.format(payment)}"
st.markdown(mes2)

#left_graph, right_graph = expander.beta_columns(2)
fig1 = interest_bar_graph(est_price, down_payment,
                          interest_rate, loan_term)
st.pyplot(fig1)

st.write('Pay off the loan sooner by adding montly payments to the principal?')

extra = st.number_input(label='Extra Monthly Payment',
                        value=0, step=100, min_value=0)
fig2 = amortization_schedule(
    est_price, down_payment, interest_rate, loan_term, extra)

st.pyplot(fig2)
