import streamlit as st
import pandas as pd
import folium
from folium import plugins
from streamlit_folium import folium_static
import time
from avm_functions import find_nearest_properties, calculate_mortgage, burndown_chart, interest_bar_graph, amortization_schedule, neighborhood_details
st.set_option('deprecation.showPyplotGlobalUse', False)

import warnings
warnings.filterwarnings('ignore')

# Create a function that will transform numeric values into currency
def prop_func(x): return '${:,.2f}'.format(x)


st.title("How much is it really going to cost to buy a house in Los Angeles?")
st.sidebar.write('<strong>Describe your ideal house</strong>',
                 unsafe_allow_html=True)

# Load in the data
neighborhoods = pd.read_csv('Data/Neighborhoods_final.csv')
houses = pd.read_csv('Data/houses.csv')

# Select box for the user
neighborhood_option = st.sidebar.selectbox(
    'Select a neighborhood you are interested in.',
    neighborhoods['neighborhood_name'])

# Creating a side bar for the house features
left_column, right_column = st.sidebar.beta_columns(2)
bedrooms = left_column.number_input(
    label="Bedrooms", format="%i", value=2, max_value=8, min_value=1)
bathrooms = right_column.number_input(
    label="Bathrooms", format="%i", value=2, max_value=8, min_value=1)
lot_size = st.sidebar.number_input(
    label="Lot Size (You can also type in a number)", format="%i", value=6500, step=250, max_value=25000)
home_size = st.sidebar.number_input(
    label="Home Size (You can also type in a number)", format="%i", value=1500, step=250, max_value=25000)

# Grab the estimated price for the house described by the user
# As well as a dataframe of similar houses
est_price, houses = find_nearest_properties(neighborhood_option, bedrooms,
                                            bathrooms, lot_size, home_size)
down_payment_default = int(est_price * .2)
expander1 = st.beta_expander('Neighborhood Details')
df = houses[['Neighborhood', 'Lot Size', 'Home Size',
             'Bedrooms', 'Bathrooms', 'Date', 'Sale Price']]
expander1.dataframe(df)
fig = neighborhood_details(neighborhood_option, lot_size, home_size)
expander1.pyplot(fig)

expander2 = st.beta_expander('Financing')
left, middle, right = expander2.beta_columns(3)
down_payment = left.number_input(
    label="Down Payment", format="%i", value=50000, max_value=5000000, step=1000)
loan_term = middle.selectbox(
    "Loan Term",  [15, 30], 1)
interest_rate = right.slider(
    label="Interest Rate",  value=2.5, max_value=8.0, step=.05)

payment = calculate_mortgage(
    est_price, down_payment, loan_term, interest_rate)

mes1 = f"""
        *20% of the estimated price: {'${:,.2f}'.format(down_payment_default)}* \n
        *You\'re interest rate will be lower the less years there are on the term*
       """
expander2.markdown(mes1)
mes2 = f"## You're expected monthly payment is {'${:,.2f}'.format(payment)}"
expander2.markdown(mes2)

left_graph, right_graph = expander2.beta_columns(2)

left_graph.write(
    'Pay off the loan sooner by adding montly payments to the principal')

extra = right_graph.number_input(label='Extra Monthly Payment',
                                 value=0, step=100, min_value=0)

schedule = amortization_schedule(
    est_price, down_payment, interest_rate, loan_term, extra)

fig1 = burndown_chart(schedule, loan_term)
left_graph.pyplot(fig1)

fig2 = interest_bar_graph(schedule)
right_graph.pyplot(fig2)

monthly_payment = schedule.loc[0, 'monthly_payments']
original_interest_payment = schedule['interest_payment'].sum()
adjusted_interest_payment = schedule.loc[schedule['new_remaining_balance']
                                         > monthly_payment, 'new_interest_payment'].sum()

if extra != 0:

    expander2.markdown(f"With an additional {prop_func(extra)} a month...")
    expander2.markdown(
        f"You will save {prop_func(original_interest_payment-adjusted_interest_payment)} in total interest paid!")
