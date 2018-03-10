
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.cluster import KMeans


# In[2]:


adv_df = pd.read_csv('advanced_stats.csv')
per_game_df = pd.read_csv('per_game_stats.csv')
stripped_pg_df = per_game_df.drop(columns=['Rk', 'Tm', 'Pos', 'G', 'MP'])
dataframe = adv_df.merge(stripped_pg_df, on = ['Player','Year'], how = 'outer')
dataframe.to_csv('all_stats.csv', index=False)


# In[3]:


OWS_centers = []
OWS_point_guards = []
OWS_sforwards = []
OWS_pforwards = []
years = []

OWS_shooting_guards = []

DWS_centers = []
DWS_pgs = []


# In[4]:


year = 1955
while year <= 2016:
    centers = dataframe[(dataframe['Year'] == year) & (dataframe['Pos'].str.contains('C'))]
    point_guards = dataframe[(dataframe['Year'] == year) & (dataframe['Pos'].str.contains('PG'))]
    shooting_guards = dataframe[(dataframe['Year'] == year) & (dataframe['Pos'].str.contains('SG'))]
    small_forwards = dataframe[(dataframe['Year'] == year) & (dataframe['Pos'].str.contains('SF'))]
    power_forwards = dataframe[(dataframe['Year'] == year) & (dataframe['Pos'].str.contains('PF'))]

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

    sample_size = 15
    rating = 'PER'
    top_centers = centers.nlargest(sample_size, rating)
    top_pgs = point_guards.nlargest(sample_size, rating)
    center_ave_DWS = top_centers['DWS'].sum()/sample_size
    point_guard_ave_DWS = top_pgs['DWS'].sum()/sample_size

    DWS_centers.append(center_ave_DWS)
    DWS_pgs.append(point_guard_ave_DWS)
    years.append(year)
    year += 1


# In[5]:

plt.subplot(2,1,1)
mpl.rcParams['figure.figsize'] = (20,10)
plt.plot(years, OWS_centers, label="Centers", linewidth=3)
plt.plot(years, OWS_point_guards, label="Point Guards", linewidth=3)
plt.plot(years, OWS_shooting_guards, label="Shooting Guards", linewidth=3)
plt.plot(years, OWS_sforwards, label="Small Forwards", linewidth=3)
plt.plot(years, OWS_pforwards, label="Power Forwards", linewidth=3)

plt.xlabel("NBA Season")
plt.ylabel("OWs among top performers")
plt.legend(loc=2)


# In[6]:

plt.subplot(2,1,2)
plt.scatter(DWS_centers, OWS_centers, s=400, c=years, cmap="Blues", label="Centers")
plt.scatter(DWS_pgs,OWS_point_guards, s=400,c=years, cmap="Oranges", label="Point Guards")
plt.xlabel("Higher DWS →")
plt.ylabel("Higher OWS →")
plt.legend(loc=2)
plt.show()


# In[7]:


np.set_printoptions(threshold=np.inf)
target_2016 = dataframe[dataframe['Year'] == 2016]
unlabelled_data = target_2016.drop(columns=['Player', 'Year','Rk', 'Tm', 'Pos', 'G', 'MP']).dropna(axis=1, how='any')

#normalize here
kmeans = KMeans(n_clusters=5).fit(unlabelled_data)
print(list(unlabelled_data))
labels = kmeans.predict(unlabelled_data)
target_2016.insert(column = 'Prediction',value = labels, loc=0)

centers = target_2016[target_2016['Pos'].str.contains('C')]
pf = target_2016[target_2016['Pos'].str.contains('PF')]
sf = target_2016[target_2016['Pos'].str.contains('SF')]
sg = target_2016[target_2016['Pos'].str.contains('SG')]
pg = target_2016[target_2016['Pos'].str.contains('PG')]

print("centers: \n" , centers[['Prediction']].mode())
print("pf: \n" , pf[['Prediction']].mode())
print("sf: \n" , sf[['Prediction']].mode())
print("sg: \n" , sg[['Prediction']].mode())
print("pg: \n" , pg[['Prediction']].mode())


# In[10]:


from sklearn.neural_network import MLPClassifier

#create and train classifier
classifier = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(15,10,5,3))
dataset = per_game_df[per_game_df['Year'] == 2016].drop(columns=['Rk', 'Tm', 'G', 'MP','Year']).dropna(axis=1, how='any')
labels = dataset['Pos']
classifier.fit(dataset.drop(columns=['Pos','Player']), labels)

#make predictions
dataset['Prediction'] = ""
for i, row in dataset.iterrows():
    player = row.drop(['Prediction', 'Pos','Player'])
    dataset.loc[i,'Prediction'] = classifier.predict([player])

print(dataset.head(100))
