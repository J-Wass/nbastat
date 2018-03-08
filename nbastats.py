import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sys
import math
from sklearn.cluster import KMeans

adv_df = pd.read_csv('advanced_stats.csv')
per_game_df = pd.read_csv('per_game_stats.csv').drop(columns=['Rk', 'Tm', 'Pos', 'G', 'MP'])
dataframe = adv_df.merge(per_game_df, on = ['Player','Year'], how = 'outer')
dataframe.to_csv('all_stats.csv', index=False)

unlabelled_data = dataframe.drop(columns=['Player', 'Rk', 'Pos', 'Tm', 'Year'])
kmeans = KMeans(n_clusters=5).fit(unlabelled_data.values)

OWS_centers = []
OWS_point_guards = []
OWS_sforwards = []
OWS_pforwards = []
OWS_shooting_guards = []
years = []

DWS_centers = []
DWS_pgs = []

#load offensive win shares for each position
year = 1950
while year <= 2018:
    centers = dataframe[(dataframe['Year'] == year) & (dataframe['Pos_x'] == 'C')]
    point_guards = dataframe[(dataframe['Year'] == year) & (dataframe['Pos_x'] == 'PG')]
    shooting_guards = dataframe[(dataframe['Year'] == year) & (dataframe['Pos_x'] == 'SG')]
    small_forwards = dataframe[(dataframe['Year'] == year) & (dataframe['Pos_x'] == 'SF')]
    power_forwards = dataframe[(dataframe['Year'] == year) & (dataframe['Pos_x'] == 'PF')]

    sample_size = 10
    rating = 'WS'
    top_centers = centers.nlargest(sample_size, rating)
    top_pgs = point_guards.nlargest(sample_size, rating)
    top_sgs = shooting_guards.nlargest(sample_size, rating)
    top_sfor = small_forwards.nlargest(sample_size, rating)
    top_pfor = power_forwards.nlargest(sample_size, rating)

    center_ave_OWS = top_centers['OWS'].sum()/sample_size
    point_guard_ave_OWS = top_pgs['OWS'].sum()/sample_size
    shooting_guard_ave_OWS = top_sgs['OWS'].sum()/sample_size
    small_forward_ave_OWS = top_sfor['OWS'].sum()/sample_size
    power_forward_ave_OWS = top_pfor['OWS'].sum()/sample_size


    OWS_centers.append(center_ave_OWS)
    OWS_point_guards.append(point_guard_ave_OWS)
    OWS_shooting_guards.append(shooting_guard_ave_OWS)
    OWS_sforwards.append(small_forward_ave_OWS)
    OWS_pforwards.append(power_forward_ave_OWS)

    sample_size = 10
    rating = 'PER'
    top_centers = centers.nlargest(sample_size, rating)
    top_pgs = point_guards.nlargest(sample_size, rating)
    center_ave_DWS = top_centers['DWS'].sum()/sample_size
    point_guard_ave_DWS = top_pgs['DWS'].sum()/sample_size

    DWS_centers.append(center_ave_DWS)
    DWS_pgs.append(point_guard_ave_DWS)
    years.append(year)
    year += 5


#plot each position
plt.subplot(2, 1, 1)
plt.plot(years, OWS_centers, label="Centers", linewidth=2)
plt.plot(years, OWS_point_guards, label="Point Guards", linewidth=2)
#plt.plot(years, OWS_shooting_guards, label="Shooting Guards", linewidth=2)
#plt.plot(years, OWS_sforwards, label="Small Forwards", linewidth=2)
#plt.plot(years, OWS_pforwards, label="Power Forwards", linewidth=2)


#graph trendlines
c = np.poly1d(np.polyfit(years, OWS_centers, 1))
pg = np.poly1d(np.polyfit(years, OWS_point_guards, 1))
sg = np.poly1d(np.polyfit(years, OWS_shooting_guards, 1))
sf = np.poly1d(np.polyfit(years, OWS_sforwards, 1))
pf = np.poly1d(np.polyfit(years, OWS_pforwards, 1))
plt.plot(years,c(years),"b:")
plt.plot(years,pg(years),"y:")
#plt.plot(years,sg(years),"g:")
#plt.plot(years,sf(years),"r:")
#plt.plot(years,pf(years),"m:")

#set up labels and legend
plt.xlabel("NBA Season")
plt.ylabel("Winshares among top performers")
plt.legend(loc=2)

#path graph DWS vs OWS
plt.subplot(2, 1, 2)
plt.scatter(DWS_centers, OWS_centers, label="Centers", linewidth=2)
plt.scatter(DWS_pgs,OWS_point_guards, label="Point Guards", linewidth=2)
c = np.poly1d(np.polyfit(DWS_centers, OWS_centers, 1))
pg = np.poly1d(np.polyfit(DWS_pgs, OWS_point_guards, 1))
plt.plot(DWS_centers, c(DWS_centers), 'b:')
plt.plot(DWS_pgs, pg(DWS_pgs), 'y:')
plt.xlabel("Higher DWS →")
plt.ylabel("Higher OWS →")
plt.show()
