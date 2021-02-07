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
    # Load the necessary dataframe
    hoods = pd.read_csv('Data/Neighborhoods_final.csv')
    # Extract the latitude and longitude of the selected neighborhood
    lat = float(hoods.loc[hoods['neighborhood_name'] ==
                          neighborhood_option, 'latitude'])
    lng = float(hoods.loc[hoods['neighborhood_name'] ==
                          neighborhood_option, 'longitude'])
    # Find the distance between the selected neighborhood and the other neighborhoods
    hoods['distance'] = np.sqrt(
        (hoods.latitude - lat)**2 + (hoods.longitude - lng)**2)

    # Sort the dataframe by descending order
    hoods.sort_values(by='distance', inplace=True)

    return list(hoods['neighborhood_name'][:8])


def find_similar_properties(neighborhood_option, bedrooms, bathrooms, lot_size, home_size, num=5.0):
    # Load the necessary dataframes
    neighborhoods = pd.read_csv('Data/Neighborhoods_final.csv')
    houses = pd.read_csv('Data/houses.csv')
    # Identify the nearest neighborhoods
    near = find_nearest_hoods(neighborhood_option)

    # Filter the house dataframe to only include the nearest neighborhoods
    houses = houses[houses.Neighborhood.isin(near)]

    # Use KMeans clustering the group thehouses with the selected features
    # The higer the number of clusters the lower number houses will be found similar
    clusters = int(float(houses.shape[0]) / num)
    kmeans = KMeans(n_clusters=clusters)

    # Identify which cluster the selected house belongs to
    houses['cluster'] = kmeans.fit_predict(
        houses[['Bedrooms', 'Bathrooms', 'LotSize', 'HomeSize']])
    # Extract the cluster number
    pred = int(kmeans.predict(
        np.array([[bedrooms, bathrooms, lot_size, home_size]])))
    # Filter the house dataframe based on the cluster assignment
    houses = houses.loc[houses['cluster'] == pred, :].reset_index(drop=True)
    # Rename the houses columns for better presentation
    houses = houses.rename(columns={'neighborhood_name': 'Neighborhood',
                                    'LotSize': 'Lot Size',
                                    'HomeSize': 'Home Size',
                                    'Bedrooms': 'Bedrooms',
                                    'Bathrooms': 'Bathrooms',
                                    #'date': 'Date',
                                    'SalePrice': 'Sale Price'})

    # Rename the index so that it isn't 0 indexed (for better presentation)
    houses.index = np.arange(1, len(houses) + 1)

    return houses

def calculate_mortgage(est_price, down_payment, loan_term, interest_rate):

    borrowed = est_price - down_payment
    ir = interest_rate / 100
    mir = (1 + ir)**(1 / 12) - 1
    total_payments = loan_term * 12
    mp = (borrowed * mir) / (1 - (1 + mir)**-total_payments)

    return mp


def find_nearest_properties(neighborhood_option, bedrooms=2, bathrooms=2, home_size=1500, lot_size=5000):
    # Load the neighborhoods dataframe
    neighborhoods = pd.read_csv('Data/Neighborhoods_final.csv')
    # load the model
    loaded_model = pickle.load(open("xgb_model.pickle.dat", "rb"))
    # Capture the order of the colunm names
    cols_when_model_builds = loaded_model.get_booster().feature_names
    # Load the fitted kmeans object
    kmeans = load_obj('kmeans_neighborhood')
    kc = kmeans.cluster_centers_
    # This  is the geocoordinates of the neighborhood selected
    neighborhoods = neighborhoods.loc[neighborhoods['neighborhood_name']
                                      == neighborhood_option]
    neighborhoods[['HomeSize', 'LotSize', 'Bedrooms', 'Bathrooms']] = [
        home_size, lot_size, bedrooms, bathrooms]
    # Capture the latitude and the longitude
    lng = neighborhoods.longitude
    lat = neighborhoods.latitude

    # Find the distance between each house and each cluster center
    map_filter = calculate_distances(neighborhoods, kc)

    # Organize the column name order to how the model has it
    neighborhoods = neighborhoods[cols_when_model_builds]

    # Predict the estimated price of the described house
    est_price = int(loaded_model.predict(neighborhoods)**2)
    clean_price = '${:,}'.format(est_price)

    # Bring in similar houses
    houses = find_similar_properties(
        neighborhood_option, bedrooms, bathrooms, lot_size, home_size, 7)

    if houses.shape[0] == 0:
        st.header(f'We cannot find any houses matching that description.')
    else:
        results = f"# Estimated Price **{clean_price}**"
        mes1 = f"We have found {houses.shape[0]} houses with a similar description near {neighborhood_option}"
        st.markdown(results)
        st.markdown(mes1)
        st.markdown('*(You may have to zoom out to see all the houses)*')

    houses['Date'] = houses['Date'].apply(
        lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').strftime('%m/%d/%Y'))
    houses['Sale Price'] = houses['Sale Price'].apply(
        lambda x: '${:,.2f}'.format(x))
    la = folium.Map(location=[lng, lat],
                    zoom_start=13,
                    min_zoom=8)
    folium.Marker(location=[lng, lat],
                  popup=neighborhood_option).add_to(la)

    for index, row in houses.iterrows():
        popup = str(index) + ") " + str(row['Sale Price'])
        folium.Marker(location=row[['Latitude', 'Longitude']],
                      popup=popup,
                      icon=folium.Icon(color='lightgray', icon='home', ico_size=(3, 3))).add_to(la)

    folium_static(la)
    return est_price, houses

def amortization_schedule(est_price, down_payment, interest_rate, loan_term, extra=0):

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

    return schedule

def burndown_chart(schedule, loan_term):

    def prop_func(x): return '${:,.2f}'.format(x)
    monthly_payment = schedule.loc[0, 'monthly_payments']

    plt.figure(figsize=(5, 3))
    plt.plot(schedule['date'], schedule['remaining_balance'],
             label='Original Terms', marker='o', markevery=72)
    plt.plot(schedule['date'], schedule['new_remaining_balance'],
             label='With Extra Payments', marker='o', markevery=72)
    xpos = list(schedule.loc[::schedule.shape[0] - 1, 'date'])
    maturity_date = xpos[1]

    # Preventing the xticks from overlapping
    df = schedule.loc[schedule['new_remaining_balance']
                      > monthly_payment, 'date']
    # Converting to better date format
    when_zero = list(df)[-1]

    if loan_term == 30:
        if df.shape[0] < 300:
            xpos.append(when_zero)
    else:
        if df.shape[0] < 168:
            xpos.append(when_zero)

    ypos = list(schedule.loc[::72, 'remaining_balance'])
    plt.xticks(xpos, rotation=35)
    y_labels = list(map(prop_func, ypos))
    plt.yticks(ypos, y_labels)
    plt.ylim(0, schedule['remaining_balance'].max() + 10000)
    plt.legend()
    plt.title('Remaining Balance on Principal', fontsize='small')
    plt.show()


def interest_bar_graph(schedule):
    monthly_payment = schedule.loc[0, 'monthly_payments']
    original_interest_payment = schedule['interest_payment'].sum()
    adjusted_interest_payment = schedule.loc[schedule['new_remaining_balance']
                                             > monthly_payment, 'new_interest_payment'].sum()

    plt.figure(figsize=(3, 2))
    yticks = ['No Additional Payment', 'Additional Payments']
    values = [original_interest_payment, adjusted_interest_payment]
    def prop_func(x): return '${:,.2f}'.format(x)
    x_labels = list(map(prop_func, values))
    ypos = np.arange(len(yticks))
    plt.yticks(ypos, yticks, fontsize=8)
    plt.xticks([original_interest_payment, adjusted_interest_payment],
               x_labels, fontsize=8, rotation=35)
    plt.barh(ypos, values)
    plt.title(
        'Total Interest Paid Based on Extra Payments to the Principal', fontsize=8)
    plt.show()


def neighborhood_details(neighborhood_option, lot_size, home_size):
    hoods = pd.read_csv('Data/Neighborhoods_final.csv')

    barWidth = .25
    lot = float(hoods.loc[hoods['neighborhood_name'] ==
                          neighborhood_option, 'LotSize'])
    home = float(hoods.loc[hoods['neighborhood_name'] ==
                           neighborhood_option, 'HomeSize'])
    # st.write(type(home))
    y1 = [lot, home]
    y2 = [lot_size, home_size]
    # Set position of bar on X axis
    r1 = np.arange(len(y1))
    r2 = [x + barWidth for x in r1]

    # Make the plot
    plt.bar(r1, y1, width=barWidth, edgecolor='white',
            label='Average in Neighborhood')
    plt.bar(r2, y2, width=barWidth, edgecolor='white', label='You\'re house')
    plt.xticks([r + barWidth - .130 for r in range(len(y1))],
               ['Lot Size', 'Home Size'])
    plt.legend()
    plt.show()
