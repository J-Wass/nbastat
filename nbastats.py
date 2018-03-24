
# coding: utf-8

# In[1]:


import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier

#load csv's to dataframes
adv_df = pd.read_csv('advanced_stats.csv')
per_game_df = pd.read_csv('per_game_stats.csv')
stripped_pg_df = per_game_df.drop(columns=['Rk', 'Tm', 'Pos', 'G', 'MP'])
dataframe = adv_df.merge(stripped_pg_df, on = ['Player','Year'], how = 'outer')
dataframe.to_csv('all_stats.csv', index=False)

#globals
OWS_centers = []
OWS_pgs = []
OWS_sfs = []
OWS_pfs = []
OWS_sgs = []
years = []

DWS_centers = []
DWS_pgs = []
DWS_sgs = []
DWS_pfs = []
DWS_sfs = []

PA3_sgs = []
PA2_sgs = []
PA3_c = []
PA2_c = []

RB_c = []
RB_pf = []
RB_sf = []
RB_sg = []
RB_pg = []


# In[2]:


mpl.rcParams['figure.figsize'] = (20,10)

#iterate dataframe and extract information for global
year = 1955
while year <= 2017:
    centers = dataframe[(dataframe['Year'] == year) & (dataframe['Pos'].str.contains('C'))]
    point_guards = dataframe[(dataframe['Year'] == year) & (dataframe['Pos'].str.contains('PG'))]
    shooting_guards = dataframe[(dataframe['Year'] == year) & (dataframe['Pos'].str.contains('SG'))]
    small_forwards = dataframe[(dataframe['Year'] == year) & (dataframe['Pos'].str.contains('SF'))]
    power_forwards = dataframe[(dataframe['Year'] == year) & (dataframe['Pos'].str.contains('PF'))]

    sample_size = 15
    rating = 'WS'

    #calulcate ast and ppg trends of top WS centers
    if(year > 1979):
        PA3_c.append(centers.nlargest(sample_size, rating)['3PA'].sum()/sample_size)
        PA2_c.append(centers.nlargest(sample_size, rating)['2PA'].sum()/sample_size)

    if(year > 1979):
        PA3_sgs.append(shooting_guards.nlargest(sample_size, rating)['3PA'].sum()/sample_size)
        PA2_sgs.append(shooting_guards.nlargest(sample_size, rating)['2PA'].sum()/sample_size)

    if(year > 1974):
        comparator = "PS/G"
        RB_c.append(centers.nlargest(sample_size, rating)[comparator].sum()/sample_size)
        RB_pf.append(power_forwards.nlargest(sample_size, rating)[comparator].sum()/sample_size)
        RB_sf.append(small_forwards.nlargest(sample_size, rating)[comparator].sum()/sample_size)
        RB_sg.append(shooting_guards.nlargest(sample_size, rating)[comparator].sum()/sample_size)
        RB_pg.append(point_guards.nlargest(sample_size, rating)[comparator].sum()/sample_size)


    top_centers = centers.nlargest(sample_size, rating)
    top_pgs = point_guards.nlargest(sample_size, rating)
    top_sgs = shooting_guards.nlargest(sample_size, rating)
    top_pfs = power_forwards.nlargest(sample_size, rating)
    top_sfs = small_forwards.nlargest(sample_size, rating)

    #calculate average OWS among top WS players
    center_ave_OWS = top_centers['OWS'].sum()/sample_size
    shooting_guard_ave_OWS = top_sgs['OWS'].sum()/sample_size
    power_forward_ave_OWS = top_pfs['OWS'].sum()/sample_size
    small_forward_ave_OWS = top_sfs['OWS'].sum()/sample_size
    point_guard_ave_OWS = top_pgs['OWS'].sum()/sample_size
    OWS_centers.append(center_ave_OWS)
    OWS_pgs.append(point_guard_ave_OWS)
    OWS_sgs.append(shooting_guard_ave_OWS)
    OWS_sfs.append(small_forward_ave_OWS)
    OWS_pfs.append(power_forward_ave_OWS)


    #calculate average DWS among top WS players
    center_ave_DWS = top_centers['DWS'].sum()/sample_size
    shooting_guard_ave_DWS = top_sgs['DWS'].sum()/sample_size
    power_forward_ave_DWS = top_pfs['DWS'].sum()/sample_size
    small_forward_ave_DWS = top_sfs['DWS'].sum()/sample_size
    point_guard_ave_DWS = top_pgs['DWS'].sum()/sample_size
    DWS_centers.append(center_ave_DWS)
    DWS_pgs.append(point_guard_ave_DWS)
    DWS_sgs.append(shooting_guard_ave_DWS)
    DWS_sfs.append(small_forward_ave_DWS)
    DWS_pfs.append(power_forward_ave_DWS)

    years.append(year)
    year += 1



# In[18]:


#plot year vs 3pa, 2pa trends among top centers
plt.subplot(1, 2, 1)
plt.axis([1980, 2017, 0, 16])
plt.title("3pt attempts vs 2pt attempts for Centers", fontsize=20)
plt.plot(years[-38:], PA3_c, label="3 pt Attempts", linewidth=3)
plt.plot(years[-38:], PA2_c, label="2 pt Attempts", linewidth=3)

z3 = np.polyfit(years[-38:], PA3_c, 1)
p3 = np.poly1d(z3)
plt.plot(years[-38:],p3(years[-38:]),"b--")

z2 = np.polyfit(years[-38:], PA2_c, 1)
p2 = np.poly1d(z2)
plt.plot(years[-38:],p2(years[-38:]),"r--")

plt.xlabel("NBA Season", fontsize=20)
plt.ylabel("Shot attempts per game", fontsize=20)
plt.legend(loc=2)

#plot year vs 3pa, 2pa trends among top guards
plt.subplot(1, 2, 2)
plt.title("3pt attempts vs 2pt attempts for Shooting Guards", fontsize=20)
plt.axis([1980, 2017, 0, 16])
plt.plot(years[-38:], PA3_sgs, label="3 pt Attempts", linewidth=3)
plt.plot(years[-38:], PA2_sgs, label="2 pt Attempts", linewidth=3)

z3 = np.polyfit(years[-38:], PA3_sgs, 1)
p3 = np.poly1d(z3)
plt.plot(years[-38:],p3(years[-38:]),"b--")

z2 = np.polyfit(years[-38:], PA2_sgs, 1)
p2 = np.poly1d(z2)
plt.plot(years[-38:],p2(years[-38:]),"r--")

plt.xlabel("NBA Season", fontsize=20)
plt.ylabel("Shot attempts per game for top SGs", fontsize=20)
plt.legend(loc=2)


# In[11]:


plt.subplot(1, 2, 1)
plt.title("Offensive Win-Share Trendlines among Top Performers", fontsize=20)

zc = np.polyfit(years, OWS_centers, 1)
pc = np.poly1d(zc)
plt.plot(years,pc(years),color="blue", label="Centers")

zpg = np.polyfit(years, OWS_pgs, 1)
ppg = np.poly1d(zpg)
plt.plot(years,ppg(years),color="orange", label="Point Guards")

zsg = np.polyfit(years, OWS_sgs, 1)
psg = np.poly1d(zsg)
plt.plot(years,psg(years),color="red", label="Shooting Guards")

zsf = np.polyfit(years, OWS_sfs, 1)
psf = np.poly1d(zsf)
plt.plot(years,psf(years),color="green", label="Small Forwards")

zpf = np.polyfit(years, OWS_pfs, 1)
ppf = np.poly1d(zpf)
plt.plot(years,ppf(years),color="purple", label="Power Forwards")


plt.xlabel("NBA Season", fontsize=20)
plt.ylabel("Average OWS among top performers", fontsize=20)
plt.legend(loc=2)

plt.subplot(1, 2, 2)
plt.title("Defensive Win-Share Trendlines among Top Performers", fontsize=20)

zc = np.polyfit(years, DWS_centers, 1)
pc = np.poly1d(zc)
plt.plot(years,pc(years),color="blue", label="Centers")

zpg = np.polyfit(years, DWS_pgs, 1)
ppg = np.poly1d(zpg)
plt.plot(years,ppg(years),color="orange", label="Point Guards")

zsg = np.polyfit(years, DWS_sgs, 1)
psg = np.poly1d(zsg)
plt.plot(years,psg(years),color="red", label="Shooting Guards")

zsf = np.polyfit(years, DWS_sfs, 1)
psf = np.poly1d(zsf)
plt.plot(years,psf(years),color="green", label="Small Forwards")

zpf = np.polyfit(years, DWS_pfs, 1)
ppf = np.poly1d(zpf)
plt.plot(years,ppf(years),color="purple", label="Power Forwards")


plt.xlabel("NBA Season", fontsize=20)
plt.ylabel("Average DWS among top performers", fontsize=20)
plt.legend(loc=2)


# In[14]:


#plot year vs OWS among top players
plt.title("Point Guard vs Center OWS (Offensive Win Shares)", fontsize=20)

plt.plot(years, OWS_centers, label="Centers", linewidth=3)
plt.plot(years, OWS_pgs, label="Point Guards", linewidth=3)

zc = np.polyfit(years, OWS_centers, 1)
pc = np.poly1d(zc)
plt.plot(years,pc(years),"b--")

zpg = np.polyfit(years, OWS_pgs, 1)
ppg = np.poly1d(zpg)
plt.plot(years,ppg(years),"--", color="orange")

plt.xlabel("NBA Season", fontsize=20)
plt.ylabel("OWS among top performers", fontsize=20)
plt.legend(loc=2)


# In[16]:


plt.title("Point Guard vs Center WS (Win Shares)", fontsize=20)

#plot DWS vs OWS among top performers
plt.scatter(DWS_centers, OWS_centers, s=400, cmap="Blues", c=years, label="Centers")
plt.scatter(DWS_pgs,OWS_pgs, s=400, cmap="Oranges", c=years, label="Point Guards")
plt.xlabel("Higher DWS →", fontsize=20)
plt.ylabel("Higher OWS →", fontsize=20)

#simple trendlines
c_x1 = sum(DWS_centers[:10])/10 #first 10
c_y1 = sum(OWS_centers[:10])/10
c_x2 = sum(DWS_centers[-10:])/10 #last 10
c_y2 = sum(OWS_centers[-10:])/10

pg_x1 = sum(DWS_pgs[:10])/10
pg_y1 = sum(OWS_pgs[:10])/10
pg_x2 = sum(DWS_pgs[-10:])/10
pg_y2 = sum(OWS_pgs[-10:])/10

plt.plot([c_x1, c_x2], [c_y1, c_y2], '--', color="blue", linewidth=3)
plt.plot([pg_x1, pg_x2], [pg_y1, pg_y2], '--', color="orange", linewidth=3)

#create legend
orange_patch = mpatches.Patch(color='orange', label='Point Guards')
blue_patch = mpatches.Patch(color='blue', label='Centers')
plt.legend(handles=[orange_patch, blue_patch], loc=2)

print("*Faded points are older seasons, opaque points are more recent seasons")


# In[15]:


#neural network for classifying player by position

#create and train classifier for 2013-2016
classifier = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(500))
dataset = dataframe[(dataframe['Year'] <= 2016) & (dataframe['Year'] >= 2000)].drop(columns=['Rk', 'Tm', 'G', 'MP','Year']).dropna(axis=1, how='any')
labels = dataset['Pos']
classifier.fit(dataset.drop(columns=['Pos','Player']), labels)

#make predictions for 2017
testing_df = dataframe[dataframe['Year'] == 2015].drop(columns=['Rk', 'Tm', 'G', 'MP','Year','PER','ORB%','DRB%', 'TRB%','AST%', 'STL%','BLK%','USG%','WS/48']).dropna(axis=1, how='any')
correct = 0
count = 0
for i, row in testing_df.iterrows():
    player = row.drop(['Pos','Player'])
    if(classifier.predict([player]) == testing_df.loc[i,'Pos']):
        correct += 1
    count += 1

print("Classifier Correct%: " , correct/count * 100)
