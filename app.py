import streamlit as st

st.title("How much is it really going to cost to buy a house?")


st.write('What is your Credit Score?')
credit_score = st.number_input('Credit Score')

st.write('How much do you have saved for a down payment?')
money_saved = st.number_input('Money Saved')

st.write('How much money do you make a year?')
salary = st.number_input('Yearly Income')
