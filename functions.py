# This module will contain all the functions requried for app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.cluster import KMeans
import folium
from folium import plugins
from streamlit_folium import folium_static
import time
import pickle
from dateutil.relativedelta import relativedelta


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


def yearsfromnow(years, from_date=None):
    if from_date is None:
        from_date = datetime.date.today()
    return from_date + relativedelta(years=years)


def monthsfromnow(months, from_date=None):
    if from_date is None:
        from_date = datetime.date.today()
    return from_date + relativedelta(months=months)


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


def find_similar_properties(neighborhood_option, bedrooms, bathrooms, lot_size, home_size, num=5.0):
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
        houses[['bedrooms', 'bathrooms', 'lot_size', 'home_size']])

    pred = int(kmeans.predict(
        np.array([[bedrooms, bathrooms, lot_size, home_size]])))

    houses = houses.loc[houses['cluster'] == pred, :]

    return houses

def calculate_mortgage(est_price, down_payment, loan_term, interest_rate):

    borrowed = est_price - down_payment
    ir = interest_rate / 100
    mir = (1 + ir)**(1 / 12) - 1
    total_payments = loan_term * 12
    mp = (borrowed * mir) / (1 - (1 + mir)**-total_payments)

    return mp


def find_nearest_properties(neighborhood_option, bedrooms=2, bathrooms=2, home_size=1500, lot_size=5000):

    neighborhoods = pd.read_csv('Data/Neighborhoods_final.csv')
    # load the model
    loaded_model = pickle.load(open("model.pickle.dat", "rb"))
    cols_when_model_builds = loaded_model.get_booster().feature_names
    # Load the fitted kmeans object
    kmeans = load_obj('kmeans_neighborhood')
    kc = kmeans.cluster_centers_
    # This  is the geocoordinates of the neighborhood selected
    neighborhoods = neighborhoods.loc[neighborhoods['neighborhood']
                                      == neighborhood_option]
    neighborhoods[['home_size', 'lot_size', 'bedrooms', 'bathrooms']] = [
        home_size, lot_size, bedrooms, bathrooms]
    lng = neighborhoods.longitude
    lat = neighborhoods.latitude
    map_filter = calculate_distances(neighborhoods, kc)
    neighborhoods = neighborhoods[cols_when_model_builds]
    neighborhoods = neighborhoods.astype(float)
    est_price = int(loaded_model.predict(neighborhoods)**2)
    clean_price = '${:,}'.format(est_price)
    # Bring in similar houses
    houses = find_similar_properties(
        neighborhood_option, bedrooms, bathrooms, lot_size, home_size, 4)

    if houses.shape[0] == 0:
        st.header(f'We cannot find any houses matching that description.')
    else:
        results = f"# Estimated Price **{clean_price}**"
        mes1 = f"We have found {houses.shape[0]} houses with a similar description near {neighborhood_option}"
        st.markdown(results)
        st.markdown(mes1)
        st.markdown('*(You may have to zoom out to see all the houses)*')

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
    return est_price

def calculate_total_interest(est_price, down_payment, interest_rate, loan_term):

    borrowed = est_price - down_payment
    interest_rate /= 100
    monthly_interest_rate = (1 + interest_rate)**(1 / 12) - 1
    total_payments = loan_term * 12
    monthly_payment = (borrowed * monthly_interest_rate) / \
        (1 - (1 + monthly_interest_rate)**-total_payments)

    first_interest_payment = borrowed * monthly_interest_rate
    first_principal_paid = monthly_payment - first_interest_payment

    schedule = pd.DataFrame({

        'payments': list(range(1, total_payments + 1)),
        'monthly_payments': monthly_payment,
        'interest_payment': first_interest_payment,
        'remaining_balance': borrowed - first_principal_paid

    })

    for i in range(total_payments - 1):
        schedule.loc[i + 1, 'interest_payment'] = schedule.loc[i,
                                                               'remaining_balance'] * monthly_interest_rate
        principal_paid = monthly_payment - \
            schedule.loc[i + 1, 'interest_payment']
        schedule.loc[i + 1, 'remaining_balance'] = schedule.loc[i,
                                                                'remaining_balance'] - principal_paid

    total_interest_paid = schedule['interest_payment'].sum()

    return total_interest_paid

def interest_bar_graph(est_price, down_payment, interest_rate, loan_term):

    interest_rate_15 = interest_rate
    interest_rate_30 = interest_rate

    if loan_term == 30:
        interest_rate_15 -= .75
    else:
        interest_rate_30 += .75

    total_15 = calculate_total_interest(
        est_price, down_payment, interest_rate_15, 15)
    total_30 = calculate_total_interest(
        est_price, down_payment, interest_rate_30, 30)

    plt.figure(figsize=(2, 1))
    yticks = ['15 year loan', '30 year loan']
    values = [total_15, total_30]
    def prop_func(x): return '${:,.2f}'.format(x)
    labels = list(map(prop_func, values))
    ypos = np.arange(len(yticks))
    plt.yticks(ypos, yticks, fontsize=5)
    plt.xticks([total_15, total_30], labels, fontsize=4)
    plt.barh(ypos, values)
    plt.title('Total Interest Paid for Different Loan Terms', fontsize=6)
    plt.show()

def amortization_schedule(est_price, down_payment, interest_rate, loan_term, extra=0):

    def prop_func(x): return '${:,.2f}'.format(x)
    borrowed = est_price - down_payment
    interest_rate /= 100
    monthly_interest_rate = (1 + interest_rate)**(1 / 12) - 1
    total_payments = loan_term * 12
    monthly_payment = (borrowed * monthly_interest_rate) / \
        (1 - (1 + monthly_interest_rate)**-total_payments)

    first_interest_payment = borrowed * monthly_interest_rate
    first_principal_paid = monthly_payment - first_interest_payment

    schedule = pd.DataFrame({

        'payments': list(range(1, total_payments + 1)),
        'monthly_payments': monthly_payment,
        'interest_payment': first_interest_payment,
        'remaining_balance': borrowed - first_principal_paid,
        'new_interest_payment': first_interest_payment,
        'new_remaining_balance': borrowed - first_principal_paid - extra

    })

    for i in range(total_payments - 1):

        # Calculate interest and principal without extra payments
        schedule.loc[i + 1, 'interest_payment'] = schedule.loc[i,
                                                               'remaining_balance'] * monthly_interest_rate
        principal_paid = monthly_payment - \
            schedule.loc[i + 1, 'interest_payment']
        schedule.loc[i + 1, 'remaining_balance'] = schedule.loc[i,
                                                                'remaining_balance'] - principal_paid

        # Calculate interest and principal with extra payments
        schedule.loc[i + 1, 'new_interest_payment'] = schedule.loc[i,
                                                                   'new_remaining_balance'] * monthly_interest_rate
        new_principal_paid = monthly_payment - \
            schedule.loc[i + 1, 'new_interest_payment'] + extra
        schedule.loc[i + 1, 'new_remaining_balance'] = schedule.loc[i,
                                                                    'new_remaining_balance'] - new_principal_paid

    schedule['date'] = datetime.date.today()

    for i in range(schedule.shape[0] - 1):
        schedule.loc[i + 1, 'date'] = monthsfromnow(1, schedule.loc[i, 'date'])

    old_total_interest_paid = schedule['interest_payment'].sum()
    new_total_interest_paid = schedule.loc[schedule['new_remaining_balance']
                                           > 0, 'new_interest_payment'].sum()

    plt.figure(figsize=(5, 3))
    plt.plot(schedule['date'], schedule['remaining_balance'],
             label='Original Terms', marker='o', markevery=72)
    plt.plot(schedule['date'], schedule['new_remaining_balance'],
             label='With Extra Payments', marker='o', markevery=72)
    xpos = list(schedule.loc[::schedule.shape[0] - 1, 'date'])
    maturity_date = xpos[1]
    df = schedule.loc[schedule['new_remaining_balance']
                      > monthly_payment, 'date']

    # Converting to better date format
    when_zero = list(df)[-1]
    #when_zero = datetime.datetime(when_zero.year, when_zero.month, when_zero.day)
    #when_zero = when_zero.strptime('%Y-%m-%d', '%m/%d/%Y')

    # Preventing the xticks from overlapping
    if loan_term == 30:
        if df.shape[0] < 300:
            xpos.append(when_zero)
    else:
        if df.shape[0] < 168:
            xpos.append(when_zero)

    ypos = list(schedule.loc[::72, 'remaining_balance'])
    plt.xticks(xpos, rotation=55)
    y_labels = list(map(prop_func, ypos))
    plt.yticks(ypos, y_labels)
    plt.ylim(0, schedule['remaining_balance'].max() + 10000)
    plt.legend()
    plt.title('Remaining Balance on Principal', fontsize='small')
    plt.show()

    st.markdown(f"With an additional {prop_func(extra)} a month...")
    st.markdown(
        f"You will save {prop_func(old_total_interest_paid-new_total_interest_paid)} in total interest paid!")
    st.markdown(
        f"You will finish paying off the loan on {when_zero} instead of {maturity_date}")
