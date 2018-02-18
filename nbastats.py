import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from nba_py import shotchart, player
from scipy.interpolate import spline

dataframe = pd.read_csv('player_stats.csv')
number_centers = []
number_point_guards = []
years = []

#load offensive win shares for each position
for year in range(1950,2018):
    df = dataframe[(dataframe['Year'] == year)]
    top_10_winshares = df.nlargest(10, 'OWS')
    number_centers.append(len(top_10_winshares[top_10_winshares['Pos'] == 'C']))
    number_point_guards.append(len(top_10_winshares[top_10_winshares['Pos'] == 'PG']))
    years.append(year)

#smooth out y values
years_smooth = np.linspace(1950,2018,10000)
guards_smooth = spline(years,number_point_guards,years_smooth)
centers_smooth = spline(years,number_centers,years_smooth)
plt.plot(years_smooth, centers_smooth, label="Centers")
plt.plot(years_smooth, guards_smooth, label="Point Guards")

#graph trendlines
guards_z = np.polyfit(years_smooth, guards_smooth, 1)
centers_z = np.polyfit(years_smooth, centers_smooth, 1)
pg = np.poly1d(guards_z)
pc = np.poly1d(centers_z)
plt.plot(years,pg(years),"r:")
plt.plot(years,pc(years),"b:")

#set up labels and legend
plt.xlabel("NBA Season")
plt.ylabel("# players in top 10 OWS")
plt.legend(loc=2)
plt.show()
